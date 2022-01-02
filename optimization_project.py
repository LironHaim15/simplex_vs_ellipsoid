import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog


class Model:
    """
    an optimization model that solves linear programming problems.
    use either 'simplex' or 'interior-point' method.
    """

    def __init__(self, method):
        self.method = method
        self.constraints = None
        self.resources = None
        self.objective = None
        self.error = None

    def set_constraints(self, c):
        self.constraints = c

    def set_resources(self, r):
        self.resources = r

    def set_objective(self, o):
        self.objective = o

    def _is_not_initialized(self):
        """
        check if the model is initialized properly with method, objective and constraints.
        :return: bool
        """
        if self.error is not None:
            return True
        elif self.method is None or self.objective is None or self.constraints is None or self.resources is None:
            self.error = 'error: model is not initialized properly'
            return True
        elif self.method != 'interior-point' and self.method != 'simplex':
            self.error = 'error: model is not initialized properly'
            return True
        else:
            return False

    def run(self, target_func='maximize'):
        """
        run the optimization
        :param target_func: 'simplex' or 'interior-point'
        :return: the results. optimized solution if exists
        """
        if self.error is not None:
            print(self.error)
            return
        if target_func == 'maximize':
            target_op = -1
        elif target_func == 'minimize':
            target_op = 1
        else:
            self.error = 'error: invalid target_fun input. please enter "maximize" or "minimize"'
            print(self.error)
            return

        print('Objective: ', self.objective)
        print('Constraints: ', self.constraints)
        print('Resources: ', self.resources)
        print()
        max_iterations = 2 ** len(self.objective)
        res = linprog(target_op * self.objective, self.constraints, self.resources, method=self.method,
                      options={'maxiter': max_iterations})  # , 'tol':3e+28})
        print(res)
        print()
        return res

    def create_klee_minty_problem(self, dim=12):
        """
        create a Klee-Minty LP problem.
        :param dim: number of dimensions/variables
        :return:updates the model
        """
        if dim < 1:
            self.error = 'error: invalid dim input. please enter a value of int dim>0'
            return
        vector = np.zeros(dim)
        constraints = []
        """create klee-minty constraints"""
        for d in range(dim):
            if d == 1:
                vector[0] = 4 * vector[0]
                vector[1] = 1
                constraints.append(vector.tolist())
                continue
            for i in range(d):
                if vector[i] == 1:
                    vector[i] *= 4
                else:
                    vector[i] *= 2
            vector[d] = 1
            constraints.append(vector.tolist())
        self.set_constraints(constraints)
        """create klee-minty objective. using the last constraint"""
        vector /= 2
        vector = vector[:dim]
        vector[dim - 1] = 1
        self.set_objective(vector)
        """create klee-minty resources for the constraints"""
        self.set_resources([5 ** (i + 1) for i in range(dim)])

    def create_random_problem(self, max_dim=6, random_dim=False):
        """
        create a RANDOM LP problem
        :param max_dim:
        :param random_dim:
        :return:
        """
        if max_dim < 1:
            self.error = 'error: invalid dim input. please enter a value of int dim>0'
            return
        if random_dim:
            dim = np.random.randint(1, max_dim + 1)
        else:
            dim = max_dim
        num_of_constraints = np.random.randint(1, dim + 1)

        """Randomly generate a feasible LP problem"""
        objective = np.random.rand(dim)
        constraints = np.random.randn(num_of_constraints, dim)
        constraints[0, :] = np.random.rand(dim) + 0.1  # make sure problem is bounded # TODO check
        resources = np.dot(constraints, np.random.rand(dim) + 0.01)

        """set models parameters"""
        self.set_constraints(constraints)
        self.set_objective(objective)
        self.set_resources(resources)

    def add_coeeficient_noise_overall(self, factor=0.1):
        """
        add noise to the constraints of the problem
        :param factor:
        :return:
        """
        if self._is_not_initialized():
            print(self.error)
            return

        noise = np.abs(factor * np.random.randn(len(self.constraints[0]), len(self.constraints[1])))
        noise += np.ones((len(self.constraints[0]), len(self.constraints[1])))
        # noise = factor * np.random.randn(len(self.constraints[1]))
        self.constraints *= noise
        # self.objective *= noise[0]

    def add_coeeficient_scale(self, factor=3):
        """
        scale the problem and alter its constraints and objective.
        it is a reduction to another problem that may be solved faster.
        :param factor:
        :return:
        """
        if self._is_not_initialized():
            print(self.error)
            return
        print(len(self.constraints[0]), len(self.constraints[1]))
        constraint_scale = np.ones((len(self.constraints[0]), len(self.constraints[1])))
        for row in constraint_scale:
            for i in range(len(row)):
                row[i] = factor ** i

        self.constraints *= constraint_scale
        self.objective *= constraint_scale[0]


def check_average_random_problems_result(method, amount=100, dim=12, random_dim=False):
    """

    :param method: 'simplex' or 'interior-point'
    :param amount: problems to solve
    :param dim: dimension of problems
    :param random_dim: bool. if want to create random problems with different dimensions
    :return:
    """
    s = Model(method)
    avg_iters = 0
    for i in range(amount):
        s.create_random_problem(max_dim=dim, random_dim=random_dim)
        result = s.run(target_func='maximize')
        avg_iters += result['nit']
    avg_iters /= amount
    print('-------------------------------------------------------------------')
    print(f'Average iterations of {amount} random problems (dim={dim}):')
    print(avg_iters)
    print('-------------------------------------------------------------------')


def check_klee_minty_problem_result(method, apply_scale, noise_factor=2, dim=12):
    """

    :param method: 'simplex' or 'interior-point'
    :param apply_scale: bool
    :param noise_factor:
    :param dim: dimension of problems
    :return:
    """
    s = Model(method)
    s.create_klee_minty_problem(dim=dim)
    if apply_scale:
        s.add_coeeficient_scale(factor=noise_factor)
    result = s.run(target_func='maximize')
    print('-------------------------------------------------------------------')
    print(f'Number of iterations for Klee-Minty problem (dim={dim}):')
    print(result['nit'])
    print('-------------------------------------------------------------------')


def create_plots(model_name, a, b, label_a, label_b):
    """
    create a plot with 2 graphs
    :param model_name: string
    :param a: dictionary for graph1
    :param b: dictionary for graph2
    :param label_a:
    :param label_b:
    :return:
    """
    plt.text(0.4, 0.4, "", fontsize=50)
    plt.xlabel('Dimension', fontsize=10)
    plt.ylabel('No. Iterations', fontsize=10)
    plt.title(f"Model {model_name} - Klee-Minty Problem")

    lists = sorted(a.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y, label=label_a)
    lists = sorted(b.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y, label=label_b)
    plt.legend()
    plt.legend()

    # plt.savefig(f'Model_{model_name}.png')
    plt.show()
    # plt.close()


if __name__ == '__main__':
    '''uncomment each test'''

    '''SIMPLEX'''
    '''Average of random problems'''
    check_average_random_problems_result(method='simplex', amount=100, dim=12, random_dim=False)
    '''Klee-Minty problem without scaling'''
    # check_klee_minty_problem_result(method='simplex', apply_scale=False, dim=12, noise_factor=2)
    '''Klee-Minty problem with scaling'''
    # check_klee_minty_problem_result(method='simplex', apply_scale=True, dim=12, noise_factor=2)

    '''INTERIOR POINTS'''
    '''Average of random problems'''
    # check_average_random_problems_result(method='interior-point', amount=100, dim=12, random_dim=False)
    '''Klee-Minty problem without scaling'''
    # check_klee_minty_problem_result(method='interior-point', apply_scale=False, dim=12, noise_factor=2)

    '''graphs - klee-minty, normal vs scaled'''
    # dims = np.arange(2,12+1)
    # print(dims)
    # klee_minty_normal ={}
    # klee_minty_scaled ={}
    # model = Model('simplex')
    # for dim in dims:
    #     # normal klee minty
    #     model.create_klee_minty_problem(dim=dim)
    #     result = model.run(target_func='maximize')
    #     klee_minty_normal[dim] = result['nit']
    #     # scale normal klee minty
    #     model.add_coeeficient_scale(factor=2)
    #     result = model.run(target_func='maximize')
    #     klee_minty_scaled[dim] = result['nit']
    # create_plots('Simplex - Normal vs Scaled', klee_minty_normal, klee_minty_scaled, 'Normal', 'Scaled')


    '''graphs - klee-minty, simplex vs interior point'''
    # dims = np.arange(2, 12 + 1)
    # print(dims)
    # klee_minty_simplex = {}
    # klee_minty_interior_point = {}
    #
    # for dim in dims:
    #     model = Model('simplex')
    #     # simplex klee minty
    #     model.create_klee_minty_problem(dim=dim)
    #     result = model.run(target_func='maximize')
    #     klee_minty_simplex[dim] = result['nit']
    #     # interior-point normal klee minty
    #     model = Model('interior-point')
    #     model.create_klee_minty_problem(dim=dim)
    #     result = model.run(target_func='maximize')
    #     klee_minty_interior_point[dim] = result['nit']
    # create_plots('Simplex - Simplex vs Interior Point', klee_minty_simplex, klee_minty_interior_point, 'Simplex',
    #              'Interior-Point')
