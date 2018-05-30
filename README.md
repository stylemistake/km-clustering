# km-clustering

Clean room implementation of the following clustering algorithms:

- KMeans
- Expectation maximization

KMeans can work with multi-dimensional datasets, while EM can only work with
two-dimensional datasets.


## Pre-requisites

- Python `^3.6`
- Python libraries:
  - `numpy`
  - `scipy`
  - `matplotlib` - for scatter plots (optional)


## Usage

```
usage: bin/clustering [-h] [-i <file>] [-o <file>] [-c <list>] [-k <number>]
                      [-d <type>] [--kmeans] [--em] [--plot] [--print]

optional arguments:
  -h, --help            show this help message and exit
  -i <file>             path to the input CSV file
  -o <file>             path to the output CSV file
  -c <list>, --columns <list>
                        specify CSV columns to use (example: 0,1,2)
  -k <number>, --clusters <number>
                        number of clusters
  -d <type>, --distance <type>
                        distance function (eucl, sad, ssd, chebyshev)
  --kmeans              use KMeans algorithm
  --em                  use Expectation Minimization algorithm
  --plot                show a scatter plot with resulting clusters
  --print               print results to console
```


## Example

Following example will clusterize the Iris dataset:

```
bin/clustering --kmeans -i data/iris.csv -c 2,3 -k 3 --plot
```

- `--kmeans` - use kmeans
- `-i data/iris.csv` - use Iris dataset
- `-c 2,3` - use columns 2 and 3 from this dataset (counting from zero)
- `-k 3` - make 3 clusters
- `--plot` - show a scatter plot


## License

This software is covered by the MIT license. See [LICENSE.md].


## Contacts

Aleksej Komarov <[stylemistake@gmail.com]>


[LICENSE.md]: LICENSE.md
[stylemistake@gmail.com]: mailto:stylemistake@gmail.com
