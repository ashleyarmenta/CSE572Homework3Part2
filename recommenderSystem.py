import matplotlib.pyplot as plt
from surprise import SVD, Dataset, KNNBasic, Reader
from surprise.model_selection import cross_validate

# Define the format of the data file (line format) for reading with Surprise
reader = Reader(line_format='userID movieID rating timestamp', sep=',', skip_lines=1)

# Load the data from the csv file
data = Dataset.load_from_file('/Users/ashleyarmenta/Downloads/archive/ratings_small.csv', reader=reader)

# Define the algorithms to use
algorithms = {
    'PMF': SVD(biased=False),  
    'UserCF': KNNBasic(sim_options={'user_based': True}),  
    'ItemCF': KNNBasic(sim_options={'user_based': False})
}

# Compute and print the MAE and RMSE for each algorithm using 5-folds cross-validation
for name, algorithm in algorithms.items():
    results = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    print(f"{name} Results:")
    print(f"Average MAE: {sum(results['test_mae']) / len(results['test_mae'])}")
    print(f"Average RMSE: {sum(results['test_rmse']) / len(results['test_rmse'])}")
    print()

# Define similarity options
similarity_options = {
    'cosine': {'name': 'cosine', 'user_based': True},
    'MSD': {'name': 'MSD', 'user_based': True},
    'pearson': {'name': 'pearson', 'user_based': True}
}

# Run the algorithms
results = {}
for name, sim_options in similarity_options.items():
    for user_based in (True, False):
        sim_options['user_based'] = user_based
        algo = KNNBasic(sim_options=sim_options)
        cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        key = f"User-based CF ({name})" if user_based else f"Item-based CF ({name})"
        results[key] = {
            'RMSE': sum(cv_results['test_rmse']) / len(cv_results['test_rmse']),
            'MAE': sum(cv_results['test_mae']) / len(cv_results['test_mae'])
        }
        print(f"{key}: Avg RMSE: {results[key]['RMSE']:.3f}, Avg MAE: {results[key]['MAE']:.3f}")

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for i, metric in enumerate(['RMSE', 'MAE']):
    axes[i].set_title(f'Average {metric}')
    axes[i].set_xlabel('Algorithm')
    axes[i].set_ylabel(metric)
    algorithms = list(results.keys())
    values = [results[algo][metric] for algo in algorithms]
    axes[i].bar(range(len(algorithms)), values)
    axes[i].set_xticks(range(len(algorithms)))
    axes[i].set_xticklabels(algorithms, rotation=45, ha='right') 

plt.tight_layout()
plt.show()

print('Starting next task...')

# Define a range of neighbor values to test
neighborhood_sizes = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Initialize dictionaries to store the results
user_based_results = {'RMSE': [], 'MAE': []}
item_based_results = {'RMSE': [], 'MAE': []}

# Run the algorithms
for k in neighborhood_sizes:
    # User-based CF
    algo_user_based = KNNBasic(k=k, sim_options={'user_based': True})
    cv_user_based = cross_validate(algo_user_based, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    user_based_results['RMSE'].append(sum(cv_user_based['test_rmse']) / len(cv_user_based['test_rmse']))
    user_based_results['MAE'].append(sum(cv_user_based['test_mae']) / len(cv_user_based['test_mae']))

    # Item-based CF
    algo_item_based = KNNBasic(k=k, sim_options={'user_based': False})
    cv_item_based = cross_validate(algo_item_based, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    item_based_results['RMSE'].append(sum(cv_item_based['test_rmse']) / len(cv_item_based['test_rmse']))
    item_based_results['MAE'].append(sum(cv_item_based['test_mae']) / len(cv_item_based['test_mae']))

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for i, metric in enumerate(['RMSE', 'MAE']):
    axes[i].set_title(f'Impact of Number of Neighbors on {metric}')
    axes[i].set_xlabel('Number of Neighbors')
    axes[i].set_ylabel(metric)
    axes[i].plot(neighborhood_sizes, user_based_results[metric], marker='o', label='User-based CF')
    axes[i].plot(neighborhood_sizes, item_based_results[metric], marker='^', label='Item-based CF')
    axes[i].legend()
    axes[i].set_xticks(neighborhood_sizes)
    axes[i].set_xticklabels(neighborhood_sizes, rotation=45)

plt.tight_layout()
plt.show()