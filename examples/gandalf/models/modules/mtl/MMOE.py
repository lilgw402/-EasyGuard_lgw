# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-03-15 16:40:30
# Modified: 2023-03-15 16:40:30
import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class Tower(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Tower, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


class MMoE(nn.Module):
    def __init__(
        self,
        input_size,
        num_experts,
        experts_out,
        experts_hidden,
        towers_hidden,
        tasks,
    ):
        super(MMoE, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.towers_hidden = towers_hidden
        self.tasks = tasks

        self.softmax = nn.Softmax(dim=1)

        self.experts = nn.ModuleList(
            [Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_experts)]
        )
        self.w_gates = nn.ParameterList(
            [nn.Parameter(torch.randn(input_size, num_experts), requires_grad=True) for i in range(self.tasks)]
        )
        self.towers = nn.ModuleList([Tower(self.experts_out, 1, self.towers_hidden) for i in range(self.tasks)])

    def forward(self, x):
        experts_o = [e(x) for e in self.experts]
        experts_o_tensor = torch.stack(experts_o)

        gates_o = [self.softmax(x @ g) for g in self.w_gates]

        tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
        tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

        final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
        return final_output


class MMoEMtl(nn.Module):
    def __init__(self, input_size, units, num_experts, num_tasks):
        super(MMoEMtl, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.expert_kernels = torch.nn.Parameter(
            torch.randn(input_size, units, num_experts, device=self.device),
            requires_grad=True,
        )
        self.gate_kernels = torch.nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(input_size, num_experts, device=self.device),
                    requires_grad=True,
                )
                for i in range(num_tasks)
            ]
        )

        self.expert_kernels_bias = torch.nn.Parameter(
            torch.randn(units, num_experts, device=self.device),
            requires_grad=True,
        )
        self.gate_kernels_bias = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.randn(num_experts, device=self.device),
                    requires_grad=True,
                )
                for i in range(num_tasks)
            ]
        )
        self.expert_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: input, (batch_size, input_size)
        expert_kernels: (input_size, units, num_experts)
        expert_kernels_bias: (units, num_experts)
        gate_kernels: (input_size, num_experts)
        gate_kernels_bias: (num_experts)
        final_outputs: output, a list len() == num_tasks, each element has shape of (batch_size, units)
        """

        gate_outputs = []
        final_outputs = []

        if self.device == "cuda":
            x = x.cuda()

        expert_outputs = torch.einsum("ab,bcd->acd", (x, self.expert_kernels))
        expert_outputs += self.expert_kernels_bias
        expert_outputs = self.expert_activation(expert_outputs)

        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = torch.einsum("ab,bc->ac", (x, gate_kernel))
            gate_output += self.gate_kernels_bias[index]
            gate_output = nn.Softmax(dim=-1)(gate_output)
            gate_outputs.append(gate_output)

        for gate_output in gate_outputs:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)
            weighted_expert_output = expert_outputs * expanded_gate_output.expand_as(expert_outputs)
            final_outputs.append(torch.sum(weighted_expert_output, 2))

        return final_outputs
