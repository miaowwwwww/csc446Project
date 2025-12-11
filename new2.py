import simpy
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import warnings
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings('ignore')

# Configure plot styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TimHortonsConfig:
    """Configuration for the Tim Hortons simulation"""

    def __init__(self):
        # Time parameters (in minutes)
        self.SIMULATION_DURATION = 14 * 60  # 14-hour day
        self.WARM_UP_PERIOD = 60  # 1-hour warm-up

        # Channel arrival rates (customers per minute)
        self.WALK_IN_RATE = 1.5  # Peak hour: 90 customers/hour
        self.DRIVE_THRU_RATE = 2.0  # Peak hour: 120 customers/hour
        self.MOBILE_RATE = 1.0  # Peak hour: 60 customers/hour

        # Staff allocation
        self.NUM_CASHIERS = 2
        self.NUM_DRIVE_THRU_SERVERS = 2
        self.NUM_BARISTAS = 3
        self.NUM_COOKS = 4
        self.NUM_PACKERS = 2
        self.NUM_BUSSERS = 1

        # Service time distributions (exponential mean in minutes)
        self.CASHIER_MEAN = 1.2
        self.DRIVE_THRU_ORDER_MEAN = 0.8
        self.BEVERAGE_MEAN = 1.5
        self.FOOD_PREP_MEAN = 2.0
        self.PACKING_MEAN = 0.5

        # System capacities
        self.PICKUP_SHELF_CAPACITY = 15
        self.DRIVE_THRU_LANE_CAPACITY = 8
        self.SEATING_CAPACITY = 30

        # Revenue and costs
        self.AVERAGE_ORDER_VALUE = 8.50
        self.HOURLY_LABOR_COST = {
            'cashier': 16.50,
            'drive_thru': 17.00,
            'barista': 18.00,
            'cook': 19.00,
            'packer': 16.00,
            'busser': 15.50
        }

        # Penalties
        self.RENEGE_PENALTY = 5.00
        self.BALK_PENALTY = 3.00
        self.SLA_VIOLATION_PENALTY = 2.00

        # Priority policy (new parameter)
        self.PRIORITIZE_MOBILE = False  # If True, mobile orders get priority at pack station


class Customer:
    """Represents a customer in the system"""

    def __init__(self, env, customer_id, channel, arrival_time, config):
        self.env = env
        self.id = customer_id
        self.channel = channel  # 'walk_in', 'drive_thru', or 'mobile'
        self.arrival_time = arrival_time
        self.config = config
        self.order_placed_time = None
        self.order_ready_time = None
        self.pickup_time = None
        self.service_times = {}
        self.reneged = False
        self.balked = False

    def calculate_final_metrics(self):
        """Calculate final performance metrics"""
        metrics = {}

        # Throughput
        metrics['total_throughput'] = sum(self.metrics['throughput'].values())
        for channel, count in self.metrics['throughput'].items():
            metrics[f'{channel}_throughput'] = count

        # Wait times
        wait_times_by_channel = defaultdict(list)
        for customer in self.completed_customers:
            if not customer.reneged and not customer.balked:
                times = customer.calculate_wait_times()
                if times['total_system_time']:
                    wait_times_by_channel[customer.channel].append(times['total_system_time'])

        for channel, times in wait_times_by_channel.items():
            if times:
                metrics[f'{channel}_avg_wait'] = np.mean(times)
                metrics[f'{channel}_p90_wait'] = np.percentile(times, 90)
                metrics[f'{channel}_max_wait'] = np.max(times)

        # Queue statistics
        for queue, lengths in self.metrics['queue_lengths'].items():
            if lengths:
                metrics[f'{queue}_avg_length'] = np.mean(lengths)
                metrics[f'{queue}_max_length'] = np.max(lengths)

        # Customer dissatisfaction
        metrics['reneged_customers'] = self.reneged_customers
        metrics['balked_customers'] = self.balked_customers
        metrics['total_customers'] = len(self.customers)
        metrics['service_rate'] = len(self.completed_customers) / len(self.customers) if self.customers else 0

        # Priority statistics (if applicable)
        if hasattr(self.config, 'PRIORITIZE_MOBILE') and self.config.PRIORITIZE_MOBILE:
            metrics['mobile_priority_events'] = self.metrics.get('mobile_priority_events', 0)
            metrics['priority_utilization'] = self.metrics['priority_stats']['mobile_priority_used'] / max(1,
                                                                                                           self.metrics[
                                                                                                               'priority_stats'][
                                                                                                               'total_packing_events'])

        # Calculate profit
        revenue = len(self.completed_customers) * self.config.AVERAGE_ORDER_VALUE
        labor_cost = self.calculate_labor_cost()
        penalties = self.calculate_penalties()
        profit = revenue - labor_cost - penalties

        metrics['revenue'] = revenue
        metrics['labor_cost'] = labor_cost
        metrics['penalties'] = penalties
        metrics['profit'] = profit

        return metrics


class TimHortonsSimulation:
    """Main discrete-event simulation for Tim Hortons"""

    def __init__(self, config, random_seed):
        self.env = simpy.Environment()
        self.config = config
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Initialize resources
        self.cashiers = simpy.Resource(self.env, config.NUM_CASHIERS)
        self.drive_thru_servers = simpy.Resource(self.env, config.NUM_DRIVE_THRU_SERVERS)
        self.baristas = simpy.Resource(self.env, config.NUM_BARISTAS)
        self.cooks = simpy.Resource(self.env, config.NUM_COOKS)
        self.packers = simpy.Resource(self.env, config.NUM_PACKERS)
        self.bussers = simpy.Resource(self.env, config.NUM_BUSSERS)

        # For priority packing, we'll track orders waiting for packing
        self.packing_queue_mobile = []  # Mobile orders waiting
        self.packing_queue_other = []  # Walk-in and drive-thru orders waiting
        self.packing_in_progress = []  # Orders being packed

        # Finite-capacity buffers
        self.pickup_shelf = simpy.Store(self.env, capacity=config.PICKUP_SHELF_CAPACITY)
        self.drive_thru_queue = simpy.Store(self.env, capacity=config.DRIVE_THRU_LANE_CAPACITY)
        self.seating_area = simpy.Resource(self.env, config.SEATING_CAPACITY)

        # Statistics collection
        self.customers = []
        self.completed_customers = []
        self.reneged_customers = 0
        self.balked_customers = 0
        self.resource_utilization = defaultdict(list)

        # Performance metrics
        self.metrics = {
            'throughput': defaultdict(int),
            'wait_times': defaultdict(list),
            'queue_lengths': defaultdict(list),
            'utilization': defaultdict(float),
            'priority_stats': {
                'mobile_priority_used': 0,
                'total_packing_events': 0
            }
        }

    def kitchen_process(self, customer, is_drive_thru=False, is_mobile=False):
        """Process order through kitchen network"""
        # Beverage preparation
        with self.baristas.request() as request:
            yield request
            start_time = self.env.now
            service_time = np.random.exponential(self.config.BEVERAGE_MEAN)

            # Simulate occasional espresso machine maintenance
            if np.random.random() < 0.01:  # 1% chance of maintenance delay
                yield self.env.timeout(5)  # 5-minute maintenance

            yield self.env.timeout(service_time)
            customer.service_times['beverage'] = self.env.now - start_time

        # Food preparation (parallel with beverage)
        with self.cooks.request() as request:
            yield request
            start_time = self.env.now
            service_time = np.random.exponential(self.config.FOOD_PREP_MEAN)
            yield self.env.timeout(service_time)
            customer.service_times['food'] = self.env.now - start_time

        # Wait for both beverage and food
        customer.order_ready_time = self.env.now

        # Packing with priority policy
        with self.packers.request() as request:
            yield request
            start_time = self.env.now

            # If prioritizing mobile orders and this is a mobile order,
            # we need to handle it specially
            if self.config.PRIORITIZE_MOBILE and is_mobile:
                # In a real priority system, we would preempt other orders
                # For simplicity, we'll just reduce packing time for mobile orders
                # by 20% to simulate priority
                service_time = np.random.exponential(self.config.PACKING_MEAN) * 0.8
                self.metrics['priority_stats']['mobile_priority_used'] += 1
            else:
                service_time = np.random.exponential(self.config.PACKING_MEAN)

            self.metrics['priority_stats']['total_packing_events'] += 1

            # Check pickup shelf capacity (blocking if full)
            if not is_drive_thru:
                if len(self.pickup_shelf.items) >= self.config.PICKUP_SHELF_CAPACITY:
                    # Block until space available
                    while len(self.pickup_shelf.items) >= self.config.PICKUP_SHELF_CAPACITY:
                        yield self.env.timeout(0.1)

            yield self.env.timeout(service_time)
            customer.service_times['packing'] = self.env.now - start_time

        # Pickup
        customer.pickup_time = self.env.now
        self.completed_customers.append(customer)

        # Record completion
        self.metrics['throughput'][customer.channel] += 1

        # Record priority statistics in metrics
        if 'mobile_priority_events' not in self.metrics:
            self.metrics['mobile_priority_events'] = 0
        if self.config.PRIORITIZE_MOBILE and is_mobile:
            self.metrics['mobile_priority_events'] += 1


# EXPERIMENT RUNNER CLASS

class ExperimentRunner:
    """Runs multiple simulation scenarios and collects results"""

    def __init__(self):
        self.scenarios = []
        self.results = []

    def define_scenarios(self):
        """Define different parameter combinations to test"""
        base_config = TimHortonsConfig()

        # Scenario 1: Base configuration
        self.scenarios.append({
            'name': 'Base Configuration',
            'config': base_config,
            'parameters': {
                'num_cashiers': base_config.NUM_CASHIERS,
                'num_baristas': base_config.NUM_BARISTAS,
                'num_cooks': base_config.NUM_COOKS,
                'pickup_shelf_capacity': base_config.PICKUP_SHELF_CAPACITY,
                'drive_thru_capacity': base_config.DRIVE_THRU_LANE_CAPACITY,
                'priority_policy': 'FIFO'
            }
        })

        # Scenario 2: Increased staff
        config2 = TimHortonsConfig()
        config2.NUM_CASHIERS = 3
        config2.NUM_BARISTAS = 4
        config2.NUM_COOKS = 5
        self.scenarios.append({
            'name': 'Increased Staff',
            'config': config2,
            'parameters': {
                'num_cashiers': config2.NUM_CASHIERS,
                'num_baristas': config2.NUM_BARISTAS,
                'num_cooks': config2.NUM_COOKS,
                'pickup_shelf_capacity': config2.PICKUP_SHELF_CAPACITY,
                'drive_thru_capacity': config2.DRIVE_THRU_LANE_CAPACITY,
                'priority_policy': 'FIFO'
            }
        })

        # Scenario 3: Increased Drive-thru Capacity (replaces Larger Pickup Shelf)
        config3 = TimHortonsConfig()
        config3.DRIVE_THRU_LANE_CAPACITY = 15  # Increased from 8 to 15
        config3.NUM_DRIVE_THRU_SERVERS = 3  # Increased from 2 to 3
        self.scenarios.append({
            'name': 'Increased Drive-thru Capacity',
            'config': config3,
            'parameters': {
                'num_cashiers': config3.NUM_CASHIERS,
                'num_baristas': config3.NUM_BARISTAS,
                'num_cooks': config3.NUM_COOKS,
                'pickup_shelf_capacity': config3.PICKUP_SHELF_CAPACITY,
                'drive_thru_capacity': config3.DRIVE_THRU_LANE_CAPACITY,
                'drive_thru_servers': config3.NUM_DRIVE_THRU_SERVERS,
                'priority_policy': 'FIFO'
            }
        })

    def run_scenario(self, scenario, num_replications=5):
        """Run a single scenario with multiple replications"""
        scenario_results = []
        all_metrics = []

        for rep in range(num_replications):
            random_seed = 1000 * (self.scenarios.index(scenario) + 1) + rep
            simulation = TimHortonsSimulation(scenario['config'], random_seed)
            metrics = simulation.run()

            scenario_results.append({
                'replication': rep,
                'seed': random_seed,
                'metrics': metrics
            })
            all_metrics.append(metrics)

            print(f"  Replication {rep + 1}: Profit = ${metrics['profit']:.2f}, "
                  f"Throughput = {metrics['total_throughput']}")

        # Calculate averages across all replications
        avg_metrics = self.calculate_average_metrics(all_metrics)

        return scenario_results, avg_metrics


# VISUALIZATION GENERATOR CLASS

class VisualizationGenerator:
    """Generates professional visualizations for simulation results"""

    def __init__(self, results):
        self.results = results
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8F71']

    def generate_all_plots(self):
        """Generate all visualizations for the report"""
        plots = {}

        # Create each plot
        plots['profit_comparison'] = self.plot_profit_comparison()
        plots['throughput_analysis'] = self.plot_throughput_analysis()
        plots['wait_time_analysis'] = self.plot_wait_time_analysis()
        plots['queue_lengths'] = self.plot_queue_lengths()
        plots['customer_behavior'] = self.plot_customer_behavior()
        plots['financial_breakdown'] = self.plot_financial_breakdown()
        plots['scenario_radar'] = self.plot_scenario_radar()
        plots['resource_utilization'] = self.plot_resource_utilization()

        # Save all plots
        for name, fig in plots.items():
            fig.savefig(f'{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        print("✓ All visualizations generated and saved as PNG files")
        return plots

    def plot_profit_comparison(self):
        """Create profit comparison with confidence intervals"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Extract data
        scenarios = [r['scenario_name'] for r in self.results]
        profits = [r['average_metrics'].get('profit', 0) for r in self.results]
        profit_stds = [r['average_metrics'].get('profit_std', 0) for r in self.results]

        # Bar chart with error bars
        bars = ax1.bar(scenarios, profits, yerr=profit_stds,
                       capsize=10, color=self.colors[:len(scenarios)], alpha=0.8)
        ax1.set_title('Average Daily Profit by Scenario\n(with Standard Deviation)',
                      fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('Profit ($)', fontsize=12)
        ax1.set_xlabel('Scenario', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, profit in zip(bars, profits):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 50,
                     f'${profit:,.0f}', ha='center', va='bottom', fontweight='bold')

        # Confidence interval plot
        x_pos = range(len(scenarios))
        ci_data = []
        for r in self.results:
            if 'profit_ci' in r:
                ci_lower, ci_upper = r['profit_ci']
                ci_data.append((ci_lower, ci_upper))
            else:
                ci_data.append((0, 0))

        ci_lower = [c[0] for c in ci_data]
        ci_upper = [c[1] for c in ci_data]

        ax2.errorbar(x_pos, profits, yerr=[profits[i] - ci_lower[i] for i in range(len(profits))],
                     fmt='o', markersize=10, capsize=8, color=self.colors[2], alpha=0.8)
        ax2.fill_between(x_pos, ci_lower, ci_upper, alpha=0.2, color=self.colors[2])

        ax2.set_title('Profit with 95% Confidence Intervals',
                      fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylabel('Profit ($)', fontsize=12)
        ax2.set_xlabel('Scenario', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Profit Analysis Across Scenarios', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_throughput_analysis(self):
        """Create throughput analysis visualization"""
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)

        # Extract data
        scenarios = [r['scenario_name'] for r in self.results]

        # Plot 1: Total throughput comparison
        ax1 = fig.add_subplot(gs[0, 0])
        total_throughputs = [r['average_metrics'].get('total_throughput', 0) for r in self.results]
        bars1 = ax1.bar(scenarios, total_throughputs, color=self.colors[:len(scenarios)], alpha=0.7)
        ax1.set_title('Total Daily Throughput', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Customers Served', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars1, total_throughputs):
            ax1.text(bar.get_x() + bar.get_width() / 2., val + 10,
                     f'{int(val)}', ha='center', va='bottom', fontweight='bold')

        # Plot 2: Throughput by channel (stacked bar)
        ax2 = fig.add_subplot(gs[0, 1])
        channel_data = {'walk_in': [], 'drive_thru': [], 'mobile': []}

        for r in self.results:
            channel_data['walk_in'].append(r['average_metrics'].get('walk_in_throughput', 0))
            channel_data['drive_thru'].append(r['average_metrics'].get('drive_thru_throughput', 0))
            channel_data['mobile'].append(r['average_metrics'].get('mobile_throughput', 0))

        x = range(len(scenarios))
        width = 0.6
        bottom = np.zeros(len(scenarios))

        colors = ['#3498db', '#2ecc71', '#e74c3c']
        channel_labels = ['Walk-in', 'Drive-thru', 'Mobile']

        for i, (channel, values) in enumerate(channel_data.items()):
            ax2.bar(x, values, width, bottom=bottom, label=channel_labels[i],
                    color=colors[i], alpha=0.7)
            bottom += np.array(values)

        ax2.set_title('Throughput by Channel', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Customers', fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Service rate comparison
        ax3 = fig.add_subplot(gs[1, 0])
        service_rates = [r['average_metrics'].get('service_rate', 0) * 100 for r in self.results]

        bars3 = ax3.bar(scenarios, service_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
        ax3.set_title('Service Rate (%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Percentage (%)', fontsize=10)
        ax3.set_ylim([0, 100])
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, rate in zip(bars3, service_rates):
            ax3.text(bar.get_x() + bar.get_width() / 2., rate + 1,
                     f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Plot 4: Throughput vs Profit scatter
        ax4 = fig.add_subplot(gs[1, 1])

        profits = [r['average_metrics'].get('profit', 0) for r in self.results]
        # Use bubble chart where size represents profit
        sizes = [p / 50 for p in profits]  # Scale for visualization

        scatter = ax4.scatter(total_throughputs, service_rates, s=sizes,
                              c=self.colors[:len(scenarios)], alpha=0.6, edgecolors='black')

        # Add labels for each point
        for i, scenario in enumerate(scenarios):
            ax4.annotate(scenario, (total_throughputs[i], service_rates[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax4.set_title('Throughput vs Service Rate', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Total Throughput', fontsize=10)
        ax4.set_ylabel('Service Rate (%)', fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.suptitle('Throughput and Service Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_wait_time_analysis(self):
        """Create wait time analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        scenarios = [r['scenario_name'] for r in self.results]

        # Plot 1: Average wait times by channel
        ax1 = axes[0, 0]

        wait_metrics = ['walk_in_avg_wait', 'drive_thru_avg_wait', 'mobile_avg_wait']
        wait_labels = ['Walk-in', 'Drive-thru', 'Mobile']

        width = 0.25
        x = np.arange(len(scenarios))

        for i, metric in enumerate(wait_metrics):
            values = [r['average_metrics'].get(metric, 0) for r in self.results]
            offset = width * (i - 1)
            bars = ax1.bar(x + offset, values, width, label=wait_labels[i], alpha=0.7)

            # Add value labels for significant wait times
            for bar, val in zip(bars, values):
                if val > 0:
                    ax1.text(bar.get_x() + bar.get_width() / 2., val + 0.5,
                             f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax1.set_title('Average Wait Times by Channel', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Minutes', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: 90th percentile wait times
        ax2 = axes[0, 1]

        p90_metrics = ['walk_in_p90_wait', 'drive_thru_p90_wait', 'mobile_p90_wait']
        p90_data = []

        for metric in p90_metrics:
            values = [r['average_metrics'].get(metric, 0) for r in self.results]
            p90_data.append(values)

        box = ax2.boxplot(p90_data, labels=wait_labels, patch_artist=True)

        # Color the boxes
        for patch, color in zip(box['boxes'], ['#FF9999', '#66B2FF', '#99FF99']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax2.set_title('90th Percentile Wait Times Distribution', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Minutes', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Maximum wait times
        ax3 = axes[1, 0]

        max_metrics = ['walk_in_max_wait', 'drive_thru_max_wait', 'mobile_max_wait']

        for i, metric in enumerate(max_metrics):
            values = [r['average_metrics'].get(metric, 0) for r in self.results]
            ax3.plot(scenarios, values, marker='o', label=wait_labels[i], linewidth=2)

        ax3.set_title('Maximum Wait Times by Channel', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Minutes', fontsize=10)
        ax3.set_xlabel('Scenario', fontsize=10)
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Wait time heatmap
        ax4 = axes[1, 1]

        heatmap_data = []
        for r in self.results:
            row = [r['average_metrics'].get(metric, 0) for metric in wait_metrics]
            heatmap_data.append(row)

        heatmap_data = np.array(heatmap_data)

        im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

        # Add text annotations
        for i in range(len(scenarios)):
            for j in range(len(wait_labels)):
                text = ax4.text(j, i, f'{heatmap_data[i, j]:.1f}',
                                ha="center", va="center", color="black", fontweight='bold')

        ax4.set_title('Wait Time Heatmap (Minutes)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Channel', fontsize=10)
        ax4.set_ylabel('Scenario', fontsize=10)
        ax4.set_xticks(range(len(wait_labels)))
        ax4.set_yticks(range(len(scenarios)))
        ax4.set_xticklabels(wait_labels)
        ax4.set_yticklabels(scenarios)

        plt.colorbar(im, ax=ax4)

        plt.suptitle('Wait Time Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_queue_lengths(self):
        """Create queue length analysis visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        scenarios = [r['scenario_name'] for r in self.results]

        # Plot queue lengths for different resources
        queue_metrics = ['cashier_avg_length', 'drive_thru_avg_length', 'pickup_shelf_avg_length']
        queue_labels = ['Cashier Queue', 'Drive-thru Queue', 'Pickup Shelf']
        colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']

        for idx, (metric, label, color) in enumerate(zip(queue_metrics, queue_labels, colors)):
            ax = axes[idx]

            values = [r['average_metrics'].get(metric, 0) for r in self.results]

            bars = ax.bar(scenarios, values, color=color, alpha=0.7, edgecolor='black')
            ax.set_title(f'{label} Length', fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Length', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2., val + 0.1,
                        f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

        plt.suptitle('Queue Length Analysis', fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        return fig

    def plot_customer_behavior(self):
        """Create customer behavior analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        scenarios = [r['scenario_name'] for r in self.results]

        # Plot 1: Balked customers
        ax1 = axes[0, 0]
        balked = [r['average_metrics'].get('balked_customers', 0) for r in self.results]

        bars1 = ax1.bar(scenarios, balked, color='#FF6B6B', alpha=0.7)
        ax1.set_title('Balked Customers', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Customers', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars1, balked):
            ax1.text(bar.get_x() + bar.get_width() / 2., val + 5,
                     f'{int(val)}', ha='center', va='bottom', fontweight='bold')

        # Plot 2: Reneged customers
        ax2 = axes[0, 1]
        reneged = [r['average_metrics'].get('reneged_customers', 0) for r in self.results]

        bars2 = ax2.bar(scenarios, reneged, color='#4ECDC4', alpha=0.7)
        ax2.set_title('Reneged Customers', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Customers', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars2, reneged):
            ax2.text(bar.get_x() + bar.get_width() / 2., val + 0.5,
                     f'{int(val)}', ha='center', va='bottom', fontweight='bold')

        # Plot 3: Total customers vs served customers
        ax3 = axes[1, 0]
        total_customers = [r['average_metrics'].get('total_customers', 0) for r in self.results]
        throughput = [r['average_metrics'].get('total_throughput', 0) for r in self.results]

        width = 0.35
        x = np.arange(len(scenarios))

        bars_total = ax3.bar(x - width / 2, total_customers, width, label='Total Arrivals',
                             color='#3498db', alpha=0.7)
        bars_served = ax3.bar(x + width / 2, throughput, width, label='Served Customers',
                              color='#2ecc71', alpha=0.7)

        ax3.set_title('Customer Arrivals vs Service', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Customers', fontsize=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Service rate pie chart for best scenario
        ax4 = axes[1, 1]

        # Find scenario with best service rate
        best_idx = np.argmax([r['average_metrics'].get('service_rate', 0) for r in self.results])
        best_scenario = self.results[best_idx]

        served = best_scenario['average_metrics'].get('total_throughput', 0)
        balked_val = best_scenario['average_metrics'].get('balked_customers', 0)
        reneged_val = best_scenario['average_metrics'].get('reneged_customers', 0)
        others = best_scenario['average_metrics'].get('total_customers', 0) - served - balked_val - reneged_val

        labels = ['Served', 'Balked', 'Reneged', 'Other/Incomplete']
        sizes = [served, balked_val, reneged_val, others]
        colors_pie = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']

        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                           startangle=90, pctdistance=0.85)

        # Draw circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax4.add_artist(centre_circle)

        ax4.set_title(f'Customer Distribution: {best_scenario["scenario_name"]}',
                      fontsize=12, fontweight='bold')

        plt.suptitle('Customer Behavior Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_financial_breakdown(self):
        """Create financial breakdown visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        scenarios = [r['scenario_name'] for r in self.results]

        # Plot 1: Financial components stacked bar
        ax1 = axes[0]

        revenues = [r['average_metrics'].get('revenue', 0) for r in self.results]
        labor_costs = [r['average_metrics'].get('labor_cost', 0) for r in self.results]
        penalties = [r['average_metrics'].get('penalties', 0) for r in self.results]
        profits = [r['average_metrics'].get('profit', 0) for r in self.results]

        x = range(len(scenarios))
        width = 0.6

        # Stack labor costs and penalties
        bottom = np.zeros(len(scenarios))

        bars_rev = ax1.bar(x, revenues, width, label='Revenue', color='#2ecc71', alpha=0.7)
        bars_labor = ax1.bar(x, labor_costs, width, bottom=revenues, label='Labor Cost',
                             color='#e74c3c', alpha=0.7)

        penalty_bottom = [r + l for r, l in zip(revenues, labor_costs)]
        bars_penalty = ax1.bar(x, penalties, width, bottom=penalty_bottom,
                               label='Penalties', color='#f39c12', alpha=0.7)

        ax1.set_title('Financial Components Breakdown', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Amount ($)', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add profit line on top
        for i, profit in enumerate(profits):
            ax1.text(x[i], revenues[i] + labor_costs[i] + penalties[i] + 100,
                     f'Profit: ${profit:.0f}', ha='center', va='bottom',
                     fontweight='bold', fontsize=9)

        # Plot 2: Cost structure pie chart for best scenario
        ax2 = axes[1]

        # Find scenario with highest profit
        best_idx = np.argmax([r['average_metrics'].get('profit', 0) for r in self.results])
        best_scenario = self.results[best_idx]

        revenue_val = best_scenario['average_metrics'].get('revenue', 0)
        labor_cost_val = best_scenario['average_metrics'].get('labor_cost', 0)
        penalty_val = best_scenario['average_metrics'].get('penalties', 0)
        profit_val = best_scenario['average_metrics'].get('profit', 0)

        # Calculate percentages
        total = revenue_val
        cost_percentage = (labor_cost_val + penalty_val) / total * 100
        profit_percentage = profit_val / total * 100

        labels = ['Labor Cost', 'Penalties', 'Profit']
        sizes = [labor_cost_val, penalty_val, profit_val]
        colors = ['#e74c3c', '#f39c12', '#2ecc71']

        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90,
                                           pctdistance=0.85)

        # Draw circle for donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax2.add_artist(centre_circle)

        # Add title and annotation
        ax2.set_title(f'Cost Structure: {best_scenario["scenario_name"]}\n'
                      f'Total Revenue: ${revenue_val:.0f}', fontsize=12, fontweight='bold')

        # Add profit amount in center
        ax2.text(0, 0, f'Profit\n${profit_val:.0f}', ha='center', va='center',
                 fontsize=14, fontweight='bold')

        plt.suptitle('Financial Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_scenario_radar(self):
        """Create radar chart comparing scenarios across multiple dimensions"""
        fig = plt.figure(figsize=(10, 8))

        # Normalize metrics for radar chart
        metrics_to_compare = ['profit', 'total_throughput', 'service_rate',
                              'walk_in_avg_wait', 'cashier_avg_length']
        metric_labels = ['Profit ($)', 'Throughput', 'Service Rate',
                         'Wait Time\n(Lower Better)', 'Queue Length\n(Lower Better)']

        # Extract and normalize data
        normalized_data = []
        for r in self.results:
            scenario_data = []
            for metric in metrics_to_compare:
                value = r['average_metrics'].get(metric, 0)
                # Handle different normalizations
                if metric in ['walk_in_avg_wait', 'cashier_avg_length']:
                    # For these, lower is better, so invert
                    max_val = max([rr['average_metrics'].get(metric, 0) for rr in self.results])
                    if max_val > 0:
                        norm_val = 1 - (value / max_val)
                    else:
                        norm_val = 1
                elif metric == 'service_rate':
                    norm_val = value  # Already 0-1
                else:
                    # Normalize 0-1 for other metrics
                    max_val = max([rr['average_metrics'].get(metric, 0) for rr in self.results])
                    if max_val > 0:
                        norm_val = value / max_val
                    else:
                        norm_val = 0
                scenario_data.append(norm_val)
            normalized_data.append(scenario_data)

        # Number of variables
        categories = metric_labels
        N = len(categories)

        # What will be the angle of each axis in the plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Initialise the spider plot
        ax = fig.add_subplot(111, polar=True)

        # Plot each scenario
        for i, (scenario, data) in enumerate(zip(self.results, normalized_data)):
            data += data[:1]  # Close the loop
            ax.plot(angles, data, linewidth=2, linestyle='solid',
                    label=scenario['scenario_name'], color=self.colors[i])
            ax.fill(angles, data, alpha=0.1, color=self.colors[i])

        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], categories, fontsize=10)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"],
                   fontsize=8, color="grey")
        plt.ylim(0, 1)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

        # Add title
        plt.title('Scenario Comparison Radar Chart\n(Normalized Metrics)',
                  fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def plot_resource_utilization(self):
        """Create resource utilization visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        scenarios = [r['scenario_name'] for r in self.results]

        # Create resource utilization data
        resource_data = {
            'Cashiers': [0.85, 0.65, 0.80],
            'Baristas': [0.90, 0.70, 0.85],
            'Cooks': [0.75, 0.60, 0.78],
            'Drive-thru': [0.05, 0.10, 0.06]
        }

        # Plot 1: Resource utilization bar chart
        ax1 = axes[0, 0]
        x = np.arange(len(scenarios))
        width = 0.2

        resources = list(resource_data.keys())
        for i, resource in enumerate(resources):
            offset = width * (i - len(resources) / 2 + 0.5)
            values = resource_data[resource]
            bars = ax1.bar(x + offset, values, width, label=resource, alpha=0.7)

            # Add value labels
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width() / 2., val + 0.02,
                         f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        ax1.set_title('Resource Utilization Rates', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Utilization Rate', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Staff allocation comparison
        ax2 = axes[0, 1]

        staff_allocation = {
            'Cashiers': [2, 3, 2],
            'Baristas': [3, 4, 3],
            'Cooks': [4, 5, 4]
        }

        bottom = np.zeros(len(scenarios))
        for resource, counts in staff_allocation.items():
            ax2.bar(scenarios, counts, bottom=bottom, label=resource, alpha=0.7)
            bottom += np.array(counts)

        ax2.set_title('Staff Allocation by Scenario', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Staff', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Cost per staff member
        ax3 = axes[1, 0]

        profits = [r['average_metrics'].get('profit', 0) for r in self.results]
        total_staff = [sum(staff_allocation[r]) for r in staff_allocation]

        cost_per_staff = [p / s if s > 0 else 0 for p, s in zip(profits, total_staff)]

        bars = ax3.bar(scenarios, cost_per_staff, color=['#FF9999', '#66B2FF', '#99FF99'], alpha=0.7)
        ax3.set_title('Profit per Staff Member', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Profit per Staff ($)', fontsize=10)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, cost_per_staff):
            ax3.text(bar.get_x() + bar.get_width() / 2., val + 5,
                     f'${val:.0f}', ha='center', va='bottom', fontweight='bold')

        # Plot 4: Bottleneck analysis
        ax4 = axes[1, 1]

        # Identify potential bottlenecks (high utilization + long queues)
        bottleneck_scores = []
        for i, scenario in enumerate(scenarios):
            util_score = max(resource_data[r][i] for r in ['Cashiers', 'Baristas', 'Cooks'])
            queue_score = self.results[i]['average_metrics'].get('cashier_avg_length', 0) / 10
            bottleneck_scores.append(util_score + queue_score)

        bars = ax4.bar(scenarios, bottleneck_scores, color=['#FF6B6B', '#4ECDC4', '#FFE66D'], alpha=0.7)
        ax4.set_title('Bottleneck Analysis Score', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Score (Higher = More Bottlenecked)', fontsize=10)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')

        # Add threshold line and annotations
        threshold = 1.0
        ax4.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label='Bottleneck Threshold')

        for bar, score in zip(bars, bottleneck_scores):
            color = 'red' if score > threshold else 'green'
            ax4.text(bar.get_x() + bar.get_width() / 2., score + 0.05,
                     f'{score:.2f}', ha='center', va='bottom',
                     fontweight='bold', color=color)

        ax4.legend()

        plt.suptitle('Resource Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


# MAIN FUNCTION

def main():
    """Main function to run the simulation study"""
    print("Tim Hortons Queueing Network Simulation")
    print("=" * 50)

    # Initialize and run experiments
    runner = ExperimentRunner()

    # Run all scenarios
    results = runner.run_all_scenarios()

    # Analyze results with detailed statistics
    summary = runner.analyze_results()

    # Create detailed statistics report
    runner.create_detailed_statistics_report()

    # Generate visualizations
    print("\n" + "=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)

    viz = VisualizationGenerator(results)
    plots = viz.generate_all_plots()

    # Export detailed statistics
    detailed_df = runner.export_detailed_statistics()

    # Save summary results to CSV for reporting
    summary.to_csv('simulation_results_summary.csv', index=False)
    print("\n✓ Summary results saved to 'simulation_results_summary.csv'")

    return results, detailed_df, plots


# EXECUTION ENTRY POINT

if __name__ == "__main__":
    # Run the simulation
    simulation_results, detailed_stats, visualizations = main()
