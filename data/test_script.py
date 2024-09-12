import torch

# Step 1: Generate binary vectors
vectors = torch.randint(0, 2, (100, 6))

# Step 2: Count the number of '1's in each vector
ones_count = vectors.sum(dim=1)

# Step 3: Calculate the probability for each count of '1's
count_of_counts = torch.bincount(ones_count, minlength=7)
probabilities = count_of_counts.float() / vectors.size(0)

# Display the results
print("Binary Vectors:\n", vectors)
print("Count of 1's in Each Vector:\n", ones_count)
print("Occurrences of Each Count of 1's:", count_of_counts)
print("Probabilities of Each Count of 1's:", probabilities)
