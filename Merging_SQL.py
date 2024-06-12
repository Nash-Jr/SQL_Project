import json

# Load train_spider.json
with open(r"C:\Users\nacho\New folder\SQL_Project\spider\spider\train_spider.json", 'r') as f:
    train_spider = json.load(f)

# Load train_others.json
with open(r"C:\Users\nacho\New folder\SQL_Project\spider\spider\train_others.json", 'r') as f:
    train_others = json.load(f)

# Combine the two datasets
combined_train_data = train_spider + train_others

# Specify the file path where the combined dataset will be saved
output_file_path = r"C:\Users\nacho\New folder\SQL_Project\combined_train_data.json"

# Save the combined dataset to a new file
with open(output_file_path, 'w') as f:
    json.dump(combined_train_data, f)

print(f"Combined dataset contains {len(combined_train_data)} examples.")
