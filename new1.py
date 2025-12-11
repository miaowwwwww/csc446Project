
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

    def calculate_wait_times(self):
        """Calculate various wait times"""
        times = {
            'order_to_kitchen': None,
            'kitchen_to_pack': None,
            'total_system_time': None
        }

        if self.order_placed_time and self.order_ready_time:
            times['order_to_kitchen'] = self.order_ready_time - self.order_placed_time
        if self.order_ready_time and self.pickup_time:
            times['kitchen_to_pack'] = self.pickup_time - self.order_ready_time
        if self.arrival_time and self.pickup_time:
            times['total_system_time'] = self.pickup_time - self.arrival_time

        return times


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
            'utilization': defaultdict(float)
        }

    def generate_arrival_process(self, channel, rate):
        """Generate customers according to Poisson process"""
        customer_id = 0
        while True:
            # Time-varying arrival rates (simplified peak/off-peak)
            current_hour = (self.env.now % (24 * 60)) // 60
            if 7 <= current_hour <= 9 or 11 <= current_hour <= 13:  # Peak hours
                adjusted_rate = rate * 1.5
            else:
                adjusted_rate = rate * 0.7

            interarrival = np.random.exponential(1 / adjusted_rate)
            yield self.env.timeout(interarrival)

            customer_id += 1
            customer = Customer(self.env, customer_id, channel, self.env.now, self.config)
            self.customers.append(customer)

            if channel == 'walk_in':
                self.env.process(self.walk_in_process(customer))
            elif channel == 'drive_thru':
                self.env.process(self.drive_thru_process(customer))
            else:  # mobile
                self.env.process(self.mobile_process(customer))

    def walk_in_process(self, customer):
        """Process for walk-in customers"""
        # Check for balking (simplified)
        queue_length = len(self.cashiers.queue)
        if queue_length > 10:  # Arbitrary balking threshold
            customer.balked = True
            self.balked_customers += 1
            return

        # Order at cashier
        with self.cashiers.request() as request:
            yield request
            start_time = self.env.now
            customer.order_placed_time = start_time
            service_time = np.random.exponential(self.config.CASHIER_MEAN)
            yield self.env.timeout(service_time)
            customer.service_times['cashier'] = self.env.now - start_time

        # Process through kitchen
        yield from self.kitchen_process(customer)

    def drive_thru_process(self, customer):
        """Process for drive-thru customers"""
        # Check for balking based on lane capacity
        if len(self.drive_thru_queue.items) >= self.config.DRIVE_THRU_LANE_CAPACITY:
            customer.balked = True
            self.balked_customers += 1
            return

        # Enter drive-thru queue
        with self.drive_thru_queue.put(customer):
            # Order station
            with self.drive_thru_servers.request() as request:
                yield request
                start_time = self.env.now
                customer.order_placed_time = start_time
                service_time = np.random.exponential(self.config.DRIVE_THRU_ORDER_MEAN)
                yield self.env.timeout(service_time)
                customer.service_times['drive_thru_order'] = self.env.now - start_time

            # Process through kitchen
            yield from self.kitchen_process(customer, is_drive_thru=True)

    def mobile_process(self, customer):
        """Process for mobile orders"""
        # Mobile orders have promised pickup time (e.g., 15 minutes)
        promised_time = self.env.now + 15

        # Process through kitchen immediately
        customer.order_placed_time = self.env.now
        yield from self.kitchen_process(customer, is_mobile=True)

        # Check for reneging if order not ready by promised time
        if self.env.now > promised_time and not customer.order_ready_time:
            wait_time = self.env.now - promised_time
            if np.random.random() < min(wait_time / 30, 0.8):  # Increasing probability with wait
                customer.reneged = True
                self.reneged_customers += 1
                return

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

        # Packing
        with self.packers.request() as request:
            yield request
            start_time = self.env.now

            # Check pickup shelf capacity (blocking if full)
            if not is_drive_thru:
                if len(self.pickup_shelf.items) >= self.config.PICKUP_SHELF_CAPACITY:
                    # Block until space available
                    while len(self.pickup_shelf.items) >= self.config.PICKUP_SHELF_CAPACITY:
                        yield self.env.timeout(0.1)

            service_time = np.random.exponential(self.config.PACKING_MEAN)
            yield self.env.timeout(service_time)
            customer.service_times['packing'] = self.env.now - start_time

        # Pickup
        customer.pickup_time = self.env.now
        self.completed_customers.append(customer)

        # Record completion
        self.metrics['throughput'][customer.channel] += 1

    def collect_statistics(self):
        """Collect statistics during simulation"""
        while True:
            # Record queue lengths
            self.metrics['queue_lengths']['cashier'].append(len(self.cashiers.queue))
            self.metrics['queue_lengths']['drive_thru'].append(len(self.drive_thru_queue.items))
            self.metrics['queue_lengths']['pickup_shelf'].append(len(self.pickup_shelf.items))

            # Record utilization (simplified)
            yield self.env.timeout(5)  # Every 5 minutes

    def run(self):
        """Run the simulation"""
        # Start arrival processes
        self.env.process(self.generate_arrival_process('walk_in', self.config.WALK_IN_RATE))
        self.env.process(self.generate_arrival_process('drive_thru', self.config.DRIVE_THRU_RATE))
        self.env.process(self.generate_arrival_process('mobile', self.config.MOBILE_RATE))

        # Start statistics collection
        self.env.process(self.collect_statistics())

        # Run simulation
        self.env.run(until=self.config.SIMULATION_DURATION)

        return self.calculate_final_metrics()

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

    def calculate_labor_cost(self):
        """Calculate total labor cost for the day"""
        hours_worked = self.config.SIMULATION_DURATION / 60
        total_cost = 0

        for role, hourly_rate in self.config.HOURLY_LABOR_COST.items():
            if role == 'cashier':
                count = self.config.NUM_CASHIERS
            elif role == 'drive_thru':
                count = self.config.NUM_DRIVE_THRU_SERVERS
            elif role == 'barista':
                count = self.config.NUM_BARISTAS
            elif role == 'cook':
                count = self.config.NUM_COOKS
            elif role == 'packer':
                count = self.config.NUM_PACKERS
            else:  # busser
                count = self.config.NUM_BUSSERS

            total_cost += count * hourly_rate * hours_worked

        return total_cost

    def calculate_penalties(self):
        """Calculate penalties for customer dissatisfaction"""
        penalties = 0
        penalties += self.reneged_customers * self.config.RENEGE_PENALTY
        penalties += self.balked_customers * self.config.BALK_PENALTY

        # SLA violations for mobile orders
        mobile_violations = 0
        for customer in self.completed_customers:
            if customer.channel == 'mobile' and customer.order_placed_time:
                promised_time = customer.order_placed_time + 15
                if customer.pickup_time and customer.pickup_time > promised_time:
                    mobile_violations += 1

        penalties += mobile_violations * self.config.SLA_VIOLATION_PENALTY

        return penalties



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
                'pickup_shelf_capacity': base_config.PICKUP_SHELF_CAPACITY
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
                'pickup_shelf_capacity': config2.PICKUP_SHELF_CAPACITY
            }
        })

        # Scenario 3: Larger pickup shelf
        config3 = TimHortonsConfig()
        config3.PICKUP_SHELF_CAPACITY = 25
        self.scenarios.append({
            'name': 'Larger Pickup Shelf',
            'config': config3,
            'parameters': {
                'num_cashiers': config3.NUM_CASHIERS,
                'num_baristas': config3.NUM_BARISTAS,
                'num_cooks': config3.NUM_COOKS,
                'pickup_shelf_capacity': config3.PICKUP_SHELF_CAPACITY
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

    def calculate_average_metrics(self, metrics_list):
        """Calculate average metrics across multiple replications"""
        if not metrics_list:
            return {}

        avg_metrics = {}
        all_keys = set()

        # Collect all keys from all metrics dictionaries
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        # Calculate averages for each key
        for key in all_keys:
            values = []
            for metrics in metrics_list:
                if key in metrics and isinstance(metrics[key], (int, float, np.number)):
                    values.append(metrics[key])

            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
                avg_metrics[f'{key}_min'] = np.min(values)
                avg_metrics[f'{key}_max'] = np.max(values)

        return avg_metrics

    def run_all_scenarios(self):
        """Run all defined scenarios"""
        print("Running Tim Hortons Simulation Experiments")
        print("=" * 50)

        self.define_scenarios()

        for scenario in self.scenarios:
            print(f"\nScenario: {scenario['name']}")
            print(f"Parameters: {scenario['parameters']}")
            print("-" * 30)

            results, avg_metrics = self.run_scenario(scenario)

            # Calculate statistics across replications
            all_metrics = [r['metrics'] for r in results]
            df = pd.DataFrame(all_metrics)

            scenario_summary = {
                'scenario_name': scenario['name'],
                'parameters': scenario['parameters'],
                'replications': len(results),
                'detailed_results': results,
                'average_metrics': avg_metrics,
                'raw_metrics_data': all_metrics
            }

            # Calculate confidence intervals for key metrics
            key_metrics = ['profit', 'total_throughput', 'service_rate']
            for metric in key_metrics:
                if metric in df.columns:
                    data = df[metric]
                    scenario_summary[f'{metric}_mean'] = data.mean()
                    scenario_summary[f'{metric}_std'] = data.std()
                    scenario_summary[f'{metric}_ci'] = self.calculate_confidence_interval(data)

            self.results.append(scenario_summary)

            # Display average stats for this scenario
            self.display_scenario_averages(scenario_summary)

        return self.results

    def display_scenario_averages(self, scenario_summary):
        """Display average statistics for a scenario"""
        print(f"\nAverage Statistics for {scenario_summary['scenario_name']} (5 runs):")
        print("-" * 40)

        avg_metrics = scenario_summary['average_metrics']

        # Financial metrics
        print("\nFinancial Metrics:")
        print(f"  Average Daily Profit: ${avg_metrics.get('profit', 0):.2f}")
        print(f"  Std Dev of Profit: ${avg_metrics.get('profit_std', 0):.2f}")
        print(f"  Min Profit: ${avg_metrics.get('profit_min', 0):.2f}")
        print(f"  Max Profit: ${avg_metrics.get('profit_max', 0):.2f}")

        # Throughput metrics
        print("\nThroughput Metrics:")
        print(f"  Average Total Throughput: {avg_metrics.get('total_throughput', 0):.0f}")
        print(f"  Walk-in Throughput: {avg_metrics.get('walk_in_throughput', 0):.0f}")
        print(f"  Drive-thru Throughput: {avg_metrics.get('drive_thru_throughput', 0):.0f}")
        print(f"  Mobile Throughput: {avg_metrics.get('mobile_throughput', 0):.0f}")

        # Wait time metrics (if available)
        if 'walk_in_avg_wait' in avg_metrics:
            print("\nWait Time Metrics (minutes):")
            print(f"  Walk-in Avg Wait: {avg_metrics.get('walk_in_avg_wait', 0):.1f}")
            print(f"  Drive-thru Avg Wait: {avg_metrics.get('drive_thru_avg_wait', 0):.1f}")
            print(f"  Mobile Avg Wait: {avg_metrics.get('mobile_avg_wait', 0):.1f}")

        # Queue metrics
        if 'cashier_avg_length' in avg_metrics:
            print("\nQueue Metrics:")
            print(f"  Avg Cashier Queue Length: {avg_metrics.get('cashier_avg_length', 0):.1f}")
            print(f"  Avg Drive-thru Queue Length: {avg_metrics.get('drive_thru_avg_length', 0):.1f}")

        # Customer dissatisfaction
        print("\nCustomer Dissatisfaction:")
        print(f"  Average Balked Customers: {avg_metrics.get('balked_customers', 0):.0f}")
        print(f"  Average Reneged Customers: {avg_metrics.get('reneged_customers', 0):.0f}")
        print(f"  Average Service Rate: {avg_metrics.get('service_rate', 0) * 100:.1f}%")

        # Confidence intervals
        if 'profit_ci' in scenario_summary:
            ci_lower, ci_upper = scenario_summary['profit_ci']
            print(f"\n95% Confidence Interval for Profit: (${ci_lower:.2f}, ${ci_upper:.2f})")

    def calculate_confidence_interval(self, data, confidence=0.95):
        """Calculate confidence interval for a dataset"""
        n = len(data)
        if n < 2:
            return (0, 0)

        mean = np.mean(data)
        std_err = stats.sem(data)
        h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)

        return (mean - h, mean + h)

    def analyze_results(self):
        """Perform comparative analysis of all scenarios"""
        if not self.results:
            print("No results to analyze. Run scenarios first.")
            return

        print("\n" + "=" * 50)
        print("COMPARATIVE ANALYSIS - AVERAGE METRICS")
        print("=" * 50)

        # Create comprehensive comparison table
        comparison_data = []
        for result in self.results:
            avg_metrics = result['average_metrics']

            # Extract key metrics
            row = {
                'Scenario': result['scenario_name'],
                'Avg Profit ($)': f"{avg_metrics.get('profit', 0):.2f}",
                'Profit Std ($)': f"{avg_metrics.get('profit_std', 0):.2f}",
                'Avg Throughput': f"{avg_metrics.get('total_throughput', 0):.0f}",
                'Service Rate (%)': f"{avg_metrics.get('service_rate', 0) * 100:.1f}%",
                'Balked Customers': f"{avg_metrics.get('balked_customers', 0):.0f}",
                'Walk-in Wait (min)': f"{avg_metrics.get('walk_in_avg_wait', 0):.1f}" if 'walk_in_avg_wait' in avg_metrics else 'N/A',
                'Cashier Queue': f"{avg_metrics.get('cashier_avg_length', 0):.1f}" if 'cashier_avg_length' in avg_metrics else 'N/A',
                'Cashiers': result['parameters']['num_cashiers'],
                'Baristas': result['parameters']['num_baristas'],
                'Shelf Capacity': result['parameters']['pickup_shelf_capacity']
            }

            # Add confidence interval if available
            if 'profit_ci' in result:
                ci_lower, ci_upper = result['profit_ci']
                row['Profit 95% CI'] = f"(${ci_lower:.2f}, ${ci_upper:.2f})"
            else:
                row['Profit 95% CI'] = 'N/A'

            comparison_data.append(row)

        df_comparison = pd.DataFrame(comparison_data)
        print("\nComprehensive Comparison Table:")
        print(df_comparison.to_string(index=False))

        # Find optimal configuration based on multiple criteria
        print("\n" + "=" * 50)
        print("OPTIMAL CONFIGURATION ANALYSIS")
        print("=" * 50)

        # Compare scenarios across multiple dimensions
        for i, result in enumerate(self.results):
            print(f"\n{result['scenario_name']}:")
            avg_metrics = result['average_metrics']

            # Calculate metrics per staff member
            num_staff = (result['parameters']['num_cashiers'] +
                         result['parameters']['num_baristas'] +
                         result['parameters']['num_cooks'])

            profit_per_staff = avg_metrics.get('profit', 0) / num_staff
            throughput_per_staff = avg_metrics.get('total_throughput', 0) / num_staff

            print(f"  Staff Count: {num_staff}")
            print(f"  Profit per Staff: ${profit_per_staff:.2f}")
            print(f"  Throughput per Staff: {throughput_per_staff:.1f}")

        # Determine optimal based on profit
        best_profit_scenario = max(self.results,
                                   key=lambda x: x['average_metrics'].get('profit', 0))
        print(f"\n★ Optimal Configuration (Profit): {best_profit_scenario['scenario_name']}")
        print(f"  Maximum Average Profit: ${best_profit_scenario['average_metrics'].get('profit', 0):.2f}")

        return df_comparison

    def create_detailed_statistics_report(self):
        """Create detailed statistics report for all scenarios"""
        print("\n" + "=" * 60)
        print("DETAILED STATISTICAL REPORT")
        print("=" * 60)

        for result in self.results:
            print(f"\n{'=' * 40}")
            print(f"SCENARIO: {result['scenario_name']}")
            print(f"{'=' * 40}")

            # Display individual run results
            print("\nIndividual Run Results:")
            print("-" * 30)
            for rep in result['detailed_results']:
                print(f"  Run {rep['replication'] + 1} (Seed {rep['seed']}):")
                print(f"    Profit: ${rep['metrics'].get('profit', 0):.2f}")
                print(f"    Throughput: {rep['metrics'].get('total_throughput', 0)}")
                print(f"    Service Rate: {rep['metrics'].get('service_rate', 0) * 100:.1f}%")
                if 'walk_in_avg_wait' in rep['metrics']:
                    print(f"    Walk-in Wait: {rep['metrics'].get('walk_in_avg_wait', 0):.1f} min")

            # Display comprehensive averages
            avg_metrics = result['average_metrics']
            print("\nAverage Statistics (across 5 runs):")
            print("-" * 30)

            # Group metrics by category
            categories = {
                'Financial': ['profit', 'revenue', 'labor_cost', 'penalties'],
                'Throughput': ['total_throughput', 'walk_in_throughput',
                               'drive_thru_throughput', 'mobile_throughput'],
                'Wait Times': ['walk_in_avg_wait', 'drive_thru_avg_wait',
                               'mobile_avg_wait', 'walk_in_p90_wait',
                               'drive_thru_p90_wait', 'mobile_p90_wait'],
                'Queues': ['cashier_avg_length', 'drive_thru_avg_length',
                           'pickup_shelf_avg_length'],
                'Customer Experience': ['service_rate', 'balked_customers',
                                        'reneged_customers', 'total_customers']
            }

            for category, metrics in categories.items():
                print(f"\n{category}:")
                for metric in metrics:
                    if metric in avg_metrics:
                        value = avg_metrics[metric]
                        if any(x in metric for x in ['profit', 'revenue', 'cost']):
                            print(f"  {metric}: ${value:.2f}")
                        elif 'rate' in metric:
                            print(f"  {metric}: {value * 100:.1f}%")
                        elif 'wait' in metric or 'length' in metric:
                            print(f"  {metric}: {value:.2f}")
                        else:
                            print(f"  {metric}: {value:.0f}")

            # Display variability metrics
            print("\nVariability Metrics:")
            print("-" * 20)
            for metric in ['profit', 'total_throughput', 'service_rate']:
                if f'{metric}_std' in avg_metrics:
                    std = avg_metrics[f'{metric}_std']
                    mean = avg_metrics[metric]
                    if mean != 0:
                        cv = (std / mean) * 100
                        print(f"  {metric}: CV = {cv:.1f}% (Std Dev = {std:.2f})")

            print()

    def export_detailed_statistics(self, filename="detailed_statistics.csv"):
        """Export detailed statistics to CSV file"""
        if not self.results:
            print("No results to export. Run scenarios first.")
            return

        all_data = []

        for scenario_result in self.results:
            scenario_name = scenario_result['scenario_name']

            # Add individual run data
            for rep in scenario_result['detailed_results']:
                row = {
                    'Scenario': scenario_name,
                    'Replication': rep['replication'] + 1,
                    'Random_Seed': rep['seed'],
                    'Type': 'Individual_Run'
                }
                row.update(rep['metrics'])
                all_data.append(row)

            # Add average data
            avg_row = {
                'Scenario': scenario_name,
                'Replication': 'Average',
                'Random_Seed': 'N/A',
                'Type': 'Average_Across_Runs'
            }
            avg_row.update(scenario_result['average_metrics'])
            all_data.append(avg_row)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_data)

        # Reorder columns for readability
        cols = ['Scenario', 'Replication', 'Random_Seed', 'Type'] + \
               [c for c in df.columns if c not in ['Scenario', 'Replication', 'Random_Seed', 'Type']]
        df = df[cols]

        df.to_csv(filename, index=False)
        print(f"\nDetailed statistics exported to '{filename}'")

        return df

    def visualize_results(self):
        """Create visualizations of simulation results"""
        if not self.results:
            print("No results to visualize. Run scenarios first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Extract data for plotting
        scenario_names = [r['scenario_name'] for r in self.results]
        profits = [r['average_metrics'].get('profit', 0) for r in self.results]
        throughputs = [r['average_throughput'] for r in self.results]

        # Plot 1: Profit comparison
        axes[0, 0].bar(scenario_names, profits)
        axes[0, 0].set_title('Average Daily Profit by Scenario')
        axes[0, 0].set_ylabel('Profit ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Throughput comparison
        axes[0, 1].bar(scenario_names, throughputs)
        axes[0, 1].set_title('Average Daily Throughput by Scenario')
        axes[0, 1].set_ylabel('Customers Served')
        axes[0, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


 
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
