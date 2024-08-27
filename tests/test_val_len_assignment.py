from unittest import TestCase, main

import torch
from torch_linear_assignment import batch_linear_assignment_var_len_cuda
from scipy.optimize import linear_sum_assignment


class TestAssignment(TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def compare_single(self, workers, tasks, sc_workers, sc_tasks):
        self.assertEqual(workers.shape, sc_workers.shape)
        self.assertEqual(tasks.shape, sc_tasks.shape)
        self.assertTrue((workers == sc_workers).all())
        self.assertTrue((tasks == sc_tasks).all())
    
    def test_single(self):
        cost = torch.randn(2900, 200).cuda()
        assignment = batch_linear_assignment_var_len_cuda([cost])
        workers = assignment[0][0]
        tasks = assignment[0][1]

        sc_workers, sc_tasks = linear_sum_assignment(cost.cpu().numpy(), maximize=False)
        sc_workers = torch.tensor(sc_workers).cuda()
        sc_tasks = torch.tensor(sc_tasks).cuda()
        
        self.compare_single(workers, tasks, sc_workers, sc_tasks)

    def test_batch_same_shape(self):
        N = 4
        costs = list()
        for i in range(N):
            cost = torch.randn(2900, 200).cuda()
            costs.append(cost)
        assignment = batch_linear_assignment_var_len_cuda(costs)
        
        for i in range(N):
            workers = assignment[i][0]
            tasks = assignment[i][1]
            sc_workers, sc_tasks = linear_sum_assignment(costs[i].cpu().numpy(), maximize=False)
            sc_workers = torch.tensor(sc_workers).cuda()
            sc_tasks = torch.tensor(sc_tasks).cuda()
            self.compare_single(workers, tasks, sc_workers, sc_tasks)

    def test_batch_diff_shape(self):
        N = 4
        costs = list()
        costs.append(torch.randn(2900, 200).cuda())
        costs.append(torch.randn(2100, 120).cuda())
        costs.append(torch.randn(3300, 220).cuda())
        costs.append(torch.randn(1100, 240).cuda())
        assignment = batch_linear_assignment_var_len_cuda(costs)
        
        for i in range(N):
            workers = assignment[i][0]
            tasks = assignment[i][1]
            sc_workers, sc_tasks = linear_sum_assignment(costs[i].cpu().numpy(), maximize=False)
            sc_workers = torch.tensor(sc_workers).cuda()
            sc_tasks = torch.tensor(sc_tasks).cuda()
            self.compare_single(workers, tasks, sc_workers, sc_tasks)



if __name__ == "__main__":
    main()
