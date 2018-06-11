import timeit
import sys
from sklearn.datasets.samples_generator import make_blobs
from gap_statistic import OptimalK
from gap_statistic.rust import gapstat


def run_speed_test(backend):
    X, y = make_blobs(n_samples=int(1e5), n_features=5, centers=3, random_state=25)
    cluster_range = range(1, 9)
    op = OptimalK(parallel_backend='rust')
    sys.stdout.write('Starting speed test for "{}" backend....'.format(backend))
    start = timeit.default_timer()
    n_clusters = op(X, cluster_array=cluster_range)
    sys.stdout.write('...Finished in {:.4f}s w/ {} clusters\n'.format(timeit.default_timer() - start, n_clusters))


if __name__ == '__main__':
    print('-' * 30)
    print('Settting up... Using rust object @ {}:\n\n'.format(gapstat.__file__))
    for backend in ['rust', 'joblib', 'multiprocessing']:
        run_speed_test(backend)



