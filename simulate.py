# %%
from math import floor, ceil
import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms

plt.style.use("ggplot")

n = 100000

# normal distributions
noise_distribution = torch.distributions.normal.Normal(0, 1)
quality_distribution = torch.distributions.normal.Normal(0, 1)
noises = noise_distribution.sample((n,))
qualities = quality_distribution.sample((n,))
# log normal
# noise_distribution = torch.distributions.log_normal.LogNormal(0, 1)
# quality_distribution = torch.distributions.log_normal.LogNormal(0, 1)
# noises = noise_distribution.sample((n,))
# qualities = quality_distribution.sample((n,))
# power law
# inverse_cdf = lambda x, alpha: (1 - x) ** (1 / (1 - alpha))
# noises = inverse_cdf(torch.rand((n,)), 1.8)
# qualities = inverse_cdf(torch.rand((n,)), 2)



# linear
performances = noises + qualities
# multiplicative: noise is proportional to quality
# min_noise = 0
# performances = qualities + noises * torch.maximum(qualities, torch.tensor(min_noise))
# %%
# plot 2D histogram of quality vs performance
buckets = 30
plt.hist2d(qualities.cpu().numpy(), performances.cpu().numpy(), bins=buckets)
plt.xlabel("Quality")
plt.ylabel("Performance")
# %%
# plot boxplots for each bucket of what the distrib of quality is at fixed perf

# Create buckets for fixed performance
bucket_size = ((performances.max() - performances.min()) / (buckets - 1)).item()
bucket_indices = ((performances - performances.min()) / bucket_size).int()

# Group qualities by bucket
grouped_qualities = [[] for _ in range(buckets)]
for i in range(n):
    grouped_qualities[bucket_indices[i]].append(qualities[i].item())

# Remove empty buckets
positions = [i * bucket_size + performances.min().item() for i in range(buckets) if len(grouped_qualities[i]) > 0]
grouped_qualities = [bucket for bucket in grouped_qualities if len(bucket) > 0]

# Plot boxplots
plt.boxplot(grouped_qualities, positions=positions)
plt.xlabel("Performance Buckets")
plt.ylabel("Quality")

log = False
if log:
    plt.xscale("log")
    plt.yscale("log")
else:
    min_int = floor(performances.min().item())
    max_int = ceil(performances.max().item())
    max_nb_ints = 10
    if max_int - min_int > max_nb_ints:
        ticks_labels = list(range(min_int, max_int + 1, (max_int - min_int) // max_nb_ints))
    else:
        ticks_labels = list(range(min_int, max_int + 1))
    # ticks_positions = [(i - performances.min().item()) / bucket_size for i in ticks_labels]
    plt.xticks(ticks_labels, ticks_labels)

# print gray line y=x between min qual & max qual
min_qual = qualities.min().item()
max_qual = qualities.max().item()

# x_min_qual = (min_qual - performances.min().item()) / bucket_size
# x_max_qual = (max_qual - performances.min().item()) / bucket_size
# plt.plot([x_min_qual, x_max_qual], [min_qual, max_qual], color="gray")
plt.plot([min_qual, max_qual], [min_qual, max_qual], color="gray")

plt.show()
# %%
