import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import datetime

# seed the simulation
np.random.seed(42)

class Car():
    def __init__(self, road, max_speed, max_acceleration, max_deceleration, name):
        # Car properties
        self.length = 0.05
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.name = name

        # Driving
        self.max_speed = max_speed
        self.safe_distance = 0
        self.acceleration = 0.01
        self.speed = 0
        
        # behavior
        self.reaction_time = 1.5
        self.min_gap = 0.1

        # Position
        self.position = self.length + sum([car.length + car.safe_distance for car in road.cars])
        self.road = road
        self.next_road = None

        self.update_next_road()
    
    def update_next_road(self):
        if self.road.outputs:
            next_intersection = self.road.outputs[0]
            if next_intersection.outputs:
                self.next_road = np.random.choice(next_intersection.outputs, 1, replace=False)[0]
    
    def next_car_info(self):
        all_cars_ahead = [car for car in self.road.cars if car.position > self.position]
        if not all_cars_ahead:
            all_cars_ahead = [car for car in self.next_road.cars]

        if not all_cars_ahead:
            return (np.inf, self.length / 2, None)
        
        next_car = sorted(all_cars_ahead, key = lambda inter: inter.position)[0]
        car_lengths = next_car.length/2 + self.length/2

        if next_car in self.road.cars:
            distance = next_car.position - self.position
        else:
            distance = self.road.length - self.position + next_car.position

        return (distance, car_lengths, next_car.speed)
    
    def calculate_dynamic_safe_distance(self):
        """Calculate safe following distance based on current speed"""
        # Safe distance = reaction distance + braking distance + minimum gap
        reaction_distance = self.speed * self.reaction_time
        braking_distance = (self.speed ** 2) / (2 * self.max_deceleration)
        return reaction_distance + braking_distance + self.min_gap
    
    def car_following_behavior(self, next_car_distance, car_lengths, next_car_speed=None):
        """
        IDM-inspired approach
        """
        if next_car_speed is None:
            next_car_speed = self.speed  # Assume same speed if unknown
        
        safe_distance = self.calculate_dynamic_safe_distance()
        relative_speed = self.speed - next_car_speed
        critical_distance = car_lengths * 1.1
        
        # Emergency
        if next_car_distance <= critical_distance:
            return -self.max_deceleration
        
        # Calculate desired following distance
        speed_interaction_term = max(0, (self.speed * relative_speed) / 
                                   (2 * np.sqrt(self.max_acceleration * self.max_deceleration)))
        desired_distance = safe_distance + speed_interaction_term
        
        # Main IDM formula
        if next_car_distance < desired_distance:
            # Too close
            distance_ratio = desired_distance / max(next_car_distance, 0.001)  # Avoid division by zero
            free_road_term = max(0, 1 - (self.speed / self.max_speed) ** 4)
            acceleration = self.max_acceleration * (free_road_term - distance_ratio ** 2)
        else:
            # Safe distance or more
            free_road_term = 1 - (self.speed / self.max_speed) ** 4
            distance_comfort = min(1, (next_car_distance - desired_distance) / max(desired_distance, 0.001))
            acceleration = self.max_acceleration * free_road_term * distance_comfort
        
        # clip
        acceleration = max(-self.max_deceleration, min(self.max_acceleration, acceleration))
        
        return acceleration
    
    def step(self, t):
        # Get information about the next car
        next_car_distance, car_lengths, next_car_speed = self.next_car_info()

        # Limit vision to current and next road
        max_vision = self.road.length - self.position + self.next_road.length - 1
        if next_car_distance > max_vision:
            next_car_distance = np.inf

        self.acceleration = self.car_following_behavior(
            next_car_distance, car_lengths, next_car_speed
        )

        # Update speed and position
        self.speed += self.acceleration * t
        self.speed = np.clip(self.speed, a_min=0, a_max=self.max_speed)
        
        self.position += self.speed * t
        
        # Check if car has reached the end of the road
        if self.position >= self.road.length:
            over_shot = self.position - self.road.length
            
            # Remove car from current road
            if self in self.road.cars:
                self.road.cars.remove(self)
            
            # Move to next road
            self.road = self.next_road
            self.road.cars.append(self)
            self.position = over_shot
            self.update_next_road()

class Map():
    def __init__(self, scale=1, layout="random", n_cars=5):
        self.intersections = []
        self.roads = []
        self.cars = []

        # Create a map based on the layout option
        if layout == "random":
            self.random_map()
        elif layout == "cyclic":
            self.random_map(n_intersections=3, max_connections=2)
        else:
            raise ValueError("Invalid layout option chosen, choose from: 'random', 'cyclic'")

        self.add_cars(n_cars=n_cars)

    def random_map(self, n_intersections=7, max_connections=3):
        # Random coords for the intersections
        # Circle
        theta = np.linspace(0, 2 * np.pi, n_intersections + 1)
        theta = theta[:-1]
        x = np.cos(theta)
        y = np.sin(theta)

        for i, coord in enumerate(zip(x, y)):
            self.intersections.append(RoadIntersection(*coord, name=chr(i + 65), my_map=self))

        for iteration in range(max_connections):
            for intersection in self.intersections:
                other_intersections = set(self.intersections) - set([intersection])
                
                # List of all possible connections
                possible_connections = sorted(
                    [link_intersection for link_intersection in other_intersections
                    if intersection not in link_intersection.links and len(link_intersection.links) < iteration + 1],
                    key=lambda inter: inter.name
                )

                if not possible_connections:
                    continue

                connection = np.random.choice(possible_connections, 1, replace=False)[0]

                if iteration == 0:
                    intersection.link(connection)
                elif iteration == 1:
                    connection.link(intersection)
                else:
                    if np.random.uniform(low=0, high=1) < 1/3:
                        intersection.link(connection)
    
    def add_cars(self, n_cars):
        car_qualities = np.random.beta(4, 2, size=n_cars)

        max_speed_range = (0.2, 0.5)
        max_acceleration_range = (0.01, 0.05)
        max_deceleration_range = (0.01, 0.03)

        accelerations = max_acceleration_range[0] + (max_acceleration_range[1] - max_acceleration_range[0]) * car_qualities
        decelerations = max_deceleration_range[0] + (max_deceleration_range[1] - max_deceleration_range[0]) * car_qualities
        max_speeds = max_speed_range[0] + (max_speed_range[1] - max_speed_range[0]) * car_qualities
        names = [_ for _ in range(len(car_qualities))]

        car_properties = zip(max_speeds, accelerations, decelerations, names)

        for car_settings in car_properties:
            if self.roads:  # Check if roads exist
                road = np.random.choice(self.roads, 1)[0]

                car = Car(road, *car_settings)  # Reduced speed for better visibility
                road.cars.append(car)
                self.cars.append(car)

    def step(self, dt=1):
        """Advance the simulation by one time step"""
        for car in self.cars:
            car.step(dt)

    def get_visualization_data(self):
        """Get data for visualization"""
        intersection_xs = []
        intersection_ys = []
        intersection_names = []
        road_coords = []
        car_xs = []
        car_ys = []
        car_names = []

        for intersection in self.intersections:
            intersection_xs.append(intersection.x_coord)
            intersection_ys.append(intersection.y_coord)
            intersection_names.append(intersection.name)

        for road in self.roads:
            start = road.start
            end = road.end
            road_coords.append((start.x_coord, start.y_coord, 
                              end.x_coord - start.x_coord, end.y_coord - start.y_coord))
        
        for car in self.cars:
            if car.road and hasattr(car.road, 'start') and hasattr(car.road, 'end'):
                progress = min(car.position / car.road.length, 1.0)  # Clamp to [0,1]
                car_x = car.road.start.x_coord + (car.road.end.x_coord - car.road.start.x_coord) * progress
                car_y = car.road.start.y_coord + (car.road.end.y_coord - car.road.start.y_coord) * progress
                car_xs.append(car_x)
                car_ys.append(car_y)
                car_names.append(car.name)
        
        return {
            'intersections': (intersection_xs, intersection_ys, intersection_names),
            'roads': road_coords,
            'cars': (car_xs, car_ys, car_names)
        }

class RoadSegment():
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def connect_to(self, receiving_segment):
        self.outputs.append(receiving_segment)
        receiving_segment.inputs.append(self)

class FreeRoad(RoadSegment):
    def __init__(self, my_map, start=None, end=None):
        super().__init__()
        self.start = start
        self.end = end
        self.length = start.distance(end)
        self.cars = []

class RoadIntersection(RoadSegment):
    def __init__(self, x_coord, y_coord, name, my_map):
        super().__init__()
        self.links = []
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.name = name
        self.map = my_map
    
    def link(self, connection):
        self.links.append(connection)
        connection.links.append(self)
        connecting_road = FreeRoad(self.map, start=self, end=connection)
        self.connect_to(connecting_road)
        connecting_road.connect_to(connection)
        self.map.roads.append(connecting_road)
        print(f"Connected {self.name} --> {connection.name}")
    
    def distance(self, connection):
        if isinstance(connection, RoadIntersection):
            return np.clip(np.sqrt((self.x_coord - connection.x_coord) ** 2 + 
                                 (self.y_coord - connection.y_coord) ** 2), a_min=0, a_max=2)
        elif isinstance(connection, list):
            distances = []
            for con in connection:
                distances.append(self.distance(con))
            return np.array(distances)

class TrafficAnimator:
    def __init__(self, layout="cyclic", n_cars=5, dt=0.1):
        self.map = Map(layout=layout, n_cars=n_cars)
        self.dt = dt
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.set_title('Traffic Simulation', fontsize=16, fontweight='bold')
        self.ax.set_facecolor('black')

        self.intersection_scatter = self.ax.scatter([], [], s=200, c='cyan', marker='o', 
                                                  edgecolors='white', linewidth=2, zorder=3)
        self.car_scatter = self.ax.scatter([], [], s=100, c='red', marker='o', 
                                         edgecolors='white', linewidth=1, zorder=4)
        
        self.road_arrows = []
        
        self.time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                     fontsize=12, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def init_plot(self):
        """Initialize the plot with static elements"""
        for arrow in self.road_arrows:
            arrow.remove()
        self.road_arrows = []
        
        data = self.map.get_visualization_data()
        
        for road_coord in data['roads']:
            x, y, dx, dy = road_coord
            arrow = self.ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                                   arrowprops=dict(arrowstyle='->', lw=2, color='lime'))
            self.road_arrows.append(arrow)
        
        intersection_xs, intersection_ys, intersection_names = data['intersections']
        for x, y, name in zip(intersection_xs, intersection_ys, intersection_names):
            self.ax.text(x * 1.15, y * 1.15, name, ha='center', va='top', fontsize=12, 
                        fontweight='bold', color='white')
        
        return []
    
    def update_frame(self, frame):
        # Step the simulation
        self.map.step(self.dt)
        
        data = self.map.get_visualization_data()
        
        intersection_xs, intersection_ys, _ = data['intersections']
        if intersection_xs:
            self.intersection_scatter.set_offsets(np.column_stack([intersection_xs, intersection_ys]))
        
        # Update cars
        car_xs, car_ys, names = data['cars']

        if car_xs:
            self.car_scatter.set_offsets(np.column_stack([car_xs, car_ys]))
            
            # Set color based on speed
            colors = [
                'yellow' if car.speed == 0 else 
                'red' if car.acceleration < 0 else 
                'green' for car in self.map.cars
            ]
            self.car_scatter.set_color(colors)

            # Remove previous annotations if they exist
            if hasattr(self, 'car_annotations'):
                for annotation in self.car_annotations:
                    annotation.remove()

            # Add new annotations
            self.car_annotations = []
            for x, y, name in zip(car_xs, car_ys, names):
                annotation = self.ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5), ha='left', color="white")
                self.car_annotations.append(annotation)
        else:
            self.car_scatter.set_offsets(np.empty((0, 2)))

            if hasattr(self, 'car_annotations'):
                for annotation in self.car_annotations:
                    annotation.remove()
                self.car_annotations = []

        
        # Update time counter
        self.time_text.set_text(f'Time: {frame * self.dt:.1f}s\nCars: {len(car_xs)}')
        
        return [self.intersection_scatter, self.car_scatter, self.time_text]
    
    def animate(self, interval=100, frames=1000):
        """Start the animation"""
        self.init_plot()
        
        anim = animation.FuncAnimation(
            self.fig, self.update_frame, init_func=lambda: [],
            frames=frames, interval=interval, blit=False, repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        
        return anim

    def save_animation(self, anim, filename=None, file_format='mp4', fps=10, dpi=100, output_dir="animations"):
        """Save animation to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traffic_simulation_{timestamp}"
        
        # Determine file extension and writer
        if file_format.lower() == 'mp4':
            filepath = f"{filename}.mp4"
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            print(f"Saving animation as MP4 to: {filepath}")
        else:
            raise ValueError("Currently supported formats: 'mp4'")
        
        # Save the animation
        try:
            anim.save(filepath, writer=writer, dpi=dpi)
            print(f"Animation saved successfully!")
        
        except Exception as e:
            print(f"Error saving animation: {e}")

if __name__ == "__main__":
    # Create animator with cyclic layout and cars
    animator = TrafficAnimator(layout="cyclic", n_cars=40, dt=0.03)
    
    # animation
    anim = animator.animate(interval=50, frames=5000)
    
    # Saving
    animator.save_animation(anim, filename='traffic_sim', file_format='mp4', fps=180)