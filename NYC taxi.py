import csv
from typing import Tuple, List, Dict
from datetime import datetime
import time
import networkx as nx
import matplotlib.pyplot as plt
import statistics

# -----------------------------------------------------------------------------------------------------------------------
# I left some code commented, it's there to showcase the usage of functions
# functions get_time_in_minutes and unite_data are not covered in the video, but they are commented in the code
# -----------------------------------------------------------------------------------------------------------------------

def read_file(file_path: str) -> any: # function to read data from a file
    data = []
    
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            data.append(line.strip())
    return data



# Variable, which will be used throughout every task
sample_data = read_file('nyc_dataset_small.txt')

# -----------------------------------------------------------------------------------------------------------------------
# -------------------   TASK 1   ----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

# This function receives a list of number of passengers, fare amount, total amount, and tips amount
# Additionally, it receives "case" - the type of data on which we want to calculate stats.
def calculate_stats(data: List[Tuple[int, float, float, float]], case) -> Tuple[str, str, str]:
    try:
        if not data: #if list is empty, we exit the function
            return (None, None, None), (None, None, None), (None, None, None), (None, None, None)
        
        #checking the “case” and returning data accordingly
        if case == 'Fare':
            fares = [t[1] for t in data] # creating a list for fares

            # returning data in $
            return "${:.2f}".format(min(fares)), "${:.2f}".format(max(fares)), "${:.2f}".format(statistics.mean(fares))
        
        elif case == 'total':
            totals = [t[2] for t in data] # creating a list for totals

            # returning data in $
            return "${:.2f}".format(min(totals)), "${:.2f}".format(max(totals)), "${:.2f}".format(statistics.mean(totals))
        
        elif case == 'tip':
            tips = [t[3] for t in data] # creating a list for tips
            
            # returning data in $
            return "${:.2f}".format(min(tips)), "${:.2f}".format(max(tips)), "${:.2f}".format(statistics.mean(tips))
        
        elif case == 'passengers':
            passengers = [t[0] for t in data] # creating a list for passengers
            
            # returning data
            return "{:.2f}".format(min(passengers)), "{:.2f}".format(max(passengers)), "{:.2f}".format(statistics.mean(passengers))
    
    # error handling
    except Exception as e:
        print("Error:", e)
        return (None, None, None), (None, None, None), (None, None, None), (None, None, None)



# "extract_stats_from_file" recieves full data and returns number of passengers, fare amount, total amount, and tips amount
def extract_stats_from_file(sample_data):
    stats_data = [] # list for new data
    for line in sample_data:
        parts = line.split(',') # parsing "sample_data"

        # getting number of passengers, fare amount, total amount, and tips amount
        passenger_count = float(parts[3]) if parts[3] else 0.0
        fare_amount = float(parts[10])
        total_amount = float(parts[16])
        tips_amount = float(parts[13])

        # adding data to the empty list
        stats_data.append((passenger_count, fare_amount, total_amount, tips_amount))
    return stats_data





# --------------------------------------------------------------------------------

# test = extract_stats_from_file(sample_data)
# print(calculate_stats(test, 'passengers'))

# --------------------------------------------------------------------------------







#Create a function that calculate the speed of a trip in Kmh and calculate same metrics as before

# This function receives empty list and a list consisted of pickup_time, dropoff_time and distance
def get_speed(speed_data, data: List[Tuple[str, str, float]]):
    for trip in data:
        pickup_time, dropoff_time, distance = trip
        duration_seconds = (dropoff_time - pickup_time).total_seconds()
        
        if duration_seconds == 0:  # Check if duration = 0
            continue  # Skip this trip and go to the next one

        duration_hours = duration_seconds / 3600 # transform time from seconds to hours

        speed_kmh = (distance * 0.621371) / duration_hours # transforming distance from miles to kilometers and calculating km/h
        speed_data.append(round(speed_kmh)) # adding speed to the empty list

    if not speed_data:  # Check if there's data in the list
        return 0  # Return zeros if it's empty
    
    return speed_data

# Function below receives same data as function above
# This function will be used later
def get_time_in_minutes(time_data, data: List[Tuple[str, str, float]]): # Recieve time of trips in minutes
    for trip in data:
        pickup_time, dropoff_time, distance = trip 
        duration_seconds = (dropoff_time - pickup_time).total_seconds() # calculate time in seconds

        duration_minutes = duration_seconds / 60 # recalculate it in minutes

        time_data.append(round(duration_minutes)) # round the result and add it to the time_data list
        

    if not time_data:  # Check if there's data in the list
        return 0  # Return zeros if it's empty
    
    return time_data

# This function receives a list consisted of pickup_time, dropoff_time and distance
def calculate_speed(data: List[Tuple[str, str, float]]) -> Tuple[float, float, float]:
    speed_data = []
    
    speed_data = get_speed(speed_data, data) # Pass empty list and received data

    # we obtain the values ​​required by the task
    min_speed ="{} km/h".format("{:.2f}".format(min(speed_data)))
    max_speed ="{} km/h".format("{:.2f}".format(max(speed_data)))
    avg_speed ="{} km/h".format("{:.2f}".format(sum(speed_data) / len(speed_data)))

    return min_speed, max_speed, avg_speed

# It receives "sample_data" to get pickup_time, dropoff_time and distance from it
def extract_speed_from_file(sample_data):
    speed_data = []
    for line in sample_data:
        values = line.split(',')
        pickup_time = values[1]
        dropoff_time = values[2]

        pickup_time = datetime.strptime(pickup_time, "%Y-%m-%dT%H:%M:%S.%f")
        dropoff_time = datetime.strptime(dropoff_time, "%Y-%m-%dT%H:%M:%S.%f")

        distance = float(values[4])
        speed_data.append((pickup_time, dropoff_time, distance))
    return speed_data



# --------------------------------------------------------------------------------

# time_data = []
# time_data = extract_speed_from_file(sample_data)
# print(calculate_speed(time_data))

# --------------------------------------------------------------------------------








# load_zones_mapping create a dictionary of zones (zone_id: zone_name)
def load_zones_mapping(filename: str) -> Dict[int, str]:
    zones = {}
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            zone_id = int(row['LocationID'])
            zone_name = row['Zone']
            zones[zone_id] = zone_name
    return zones

# extract_pickup_zones receives "sample_data" and gets pickup locations from each trip
def extract_pickup_zones(sample_data: List[str]) -> List[int]:
    pickup_zones = []
    for line in sample_data:
        pickup_zones.append(int(line.split(',')[7]))
    return pickup_zones

# count_trips receives pickup_zones_data and zones_mapping to calculate number of trips from each pickup location
def count_trips(pickup_zones_data: List[int], zones_mapping: Dict[int, str]) -> Dict[str, int]:
    # Dictionary to store counts for each zone
    zone_counts = {}

    # Initialize counts for each zone to 0
    for zone_name in zones_mapping.values():
        zone_counts[zone_name] = 0

    # Loop through pickup zones data and update counts
    for zone_code in pickup_zones_data:
        zone_name = zones_mapping.get(zone_code)
        if zone_name in zone_counts:
            zone_counts[zone_name] += 1

    return zone_counts


# this is not commented, because it will be used later in the code

zones_filename = 'taxi+_zone_lookup.csv'

zones_mapping = load_zones_mapping(zones_filename)

# --------------------------------------------------------------------------------

# pickup_zones_data = extract_pickup_zones(sample_data)

# trip_counts = count_trips(pickup_zones_data, zones_mapping)

# print(trip_counts)

# --------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------
# -------------------   TASK 2   ----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# merge sort has a time complexity of O(n log n) in all cases, making it efficient for large datasets.
def merge_sort(arr): 
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr

# selection sort has a time complexity of O(n^2), making it less efficient compared to merge sort for large datasets
# in our case, it's 40 TIMES WORSE than merge sort
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    
    return arr

# sort_attributes returns sorted attributes using chosen algorithm
# also returns time of execution of chosen algorithm
def sort_attributes(attributes, algorithm):
    execution_time = 0 # variable to contain time of execution
    if algorithm == 'merge_sort':
        start_time = time.perf_counter()
        merge_sort(attributes)
        end_time = time.perf_counter()
        execution_time = end_time - start_time # final time of execution

    elif algorithm == 'selection_sort':
        start_time = time.perf_counter()
        selection_sort(attributes)
        end_time = time.perf_counter()
        execution_time = end_time - start_time # final time of execution
    else:
        print("Invalid algorithm choice")
        return attributes
    
    return attributes, execution_time

# unite_data puts data from stats, speed and time into one tuple
def unite_data(stats_data, speed_data, time_data):
    try:
        full_data = []
        for i in range(len(stats_data)):
            full_data.append((stats_data[i][0], stats_data[i][1], stats_data[i][2], stats_data[i][3], speed_data[i], time_data[i]))
    except Exception as e:
        print("Error:", e)
    return full_data

# choose_attributes lets you to choose what data you want to sort
def choose_attributes(attributes):
    attribute_index = input('What do you want to sort?\n1. Number of passengers \n2. fare amount \n3. total amount \n4. tips amount \n5. speed \n6. time \n\nType a number from 1 to 6:\n')
    algorithm_key = input('\nWhich algorithm you want to use? \n1. Merge sort \n2. Selection sort \n\nType a number 1/2:\n')
    algorithm = ''
    
    attribute_index = int(attribute_index) - 1

    if algorithm_key == '1':
        algorithm = 'merge_sort'
    else:
        algorithm = 'selection_sort'

    list_to_sort = [] # create empty list and fill it with chosen data to be sorted
    for tuple in attributes:
        list_to_sort.append(tuple[attribute_index])

    # pass data and algorithm of sorting. Get result and execution time
    result, execution_time = sort_attributes(list_to_sort, algorithm)
    print('Execution time = ', execution_time)
    return result



# --------------------------------------------------------------------------------

# stats_data = []
# speed_data = []
# time_data = []

# # I used functions from previous task
# stats_data = extract_stats_from_file(sample_data)
# time_and_distance = extract_speed_from_file(sample_data)
# speed_data = get_speed(speed_data, time_and_distance)
# time_data = get_time_in_minutes(time_data, time_and_distance)

# full_data = unite_data(stats_data, speed_data, time_data)
# print(choose_attributes(full_data))

# --------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------
# -------------------   TASK 3   ----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

# get pickup_and_dropoff to use as points
def get_pickup_and_dropoff(data: List[str], zones_mapping) -> List[int]:
    zones = []

    for line in data:
        zones.append((zones_mapping.get(int(line.split(',')[7])), zones_mapping.get(int(line.split(',')[8]))))
    
    return zones

def build_trip_graph(trip_data):
    G = nx.Graph()
    
    # Count the number of trips between each pair of locations
    trip_counts = {}
    for pickup, dropoff in trip_data:
        if (pickup, dropoff) in trip_counts:
            trip_counts[(pickup, dropoff)] += 1
        else:
            trip_counts[(pickup, dropoff)] = 1
    
    # Add nodes and weighted edges to the graph
    for (pickup, dropoff), count in trip_counts.items():
        G.add_edge(pickup, dropoff, weight=count)
    
    return G

def find_connected_components(trip_graph, traversal_type='dfs'):
    connected_components = []

    # Set up traversal function based on traversal type
    traversal_func = dfs_traversal if traversal_type == 'dfs' else bfs_traversal

    visited = set()
    for node in trip_graph.nodes():
        if node not in visited:
            # Start traversal from this node
            connected_component = set()
            traversal_func(trip_graph, node, visited, connected_component)
            connected_components.append(connected_component)

    return connected_components

def draw_trip_graph(trip_graph):
    pos = nx.spring_layout(trip_graph)  # Positions for all nodes
    
    # Draw nodes
    nx.draw_networkx_nodes(trip_graph, pos, node_size=50)
    
    # Draw edges with weights
    edge_weights = nx.get_edge_attributes(trip_graph, 'weight')
    nx.draw_networkx_edges(trip_graph, pos, width=[w/2 for w in edge_weights.values()])
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(trip_graph, pos, edge_labels=edge_weights)
    
    # Draw labels for nodes
    nx.draw_networkx_labels(trip_graph, pos)
    
    # Display the graph
    plt.title("Trip Graph")
    plt.axis('off')
    plt.show()

def dfs_traversal(graph, start_node, visited, connected_component):
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            connected_component.add(node)
            stack.extend(graph.neighbors(node))

def bfs_traversal(graph, start_node, visited, connected_component):
    queue = [start_node]

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            connected_component.add(node)
            queue.extend(graph.neighbors(node))


# --------------------------------------------------------------------------------

# edge_list = get_pickup_and_dropoff(sample_data, zones_mapping)
# trip_graph = build_trip_graph(edge_list)

# print(find_connected_components(trip_graph))
# draw_trip_graph(trip_graph)

# --------------------------------------------------------------------------------