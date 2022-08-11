import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, student_n, exer_n, knowledge_n):
        super(Net, self).__init__()
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.hidden_dim1, self.hidden_dim2, self.hidden_dim3= 100, 100, 100
        self.gamma = torch.nn.Linear(self.hidden_dim3, 1)
        self.nu = torch.nn.Linear(self.hidden_dim3, 1)
        self.alpha = torch.nn.Linear(self.hidden_dim3, 1)
        self.beta = torch.nn.Linear(self.hidden_dim3, 1)
        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.activate = nn.LeakyReLU(0.01)
        self.drop = nn.Dropout(p=0.5)

        self.fully1 = torch.nn.Linear(self.input_dim, self.hidden_dim1)
        self.fully2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fully3 = torch.nn.Linear(self.hidden_dim2, self.hidden_dim3)


        self.uc1 = torch.nn.Linear(self.input_dim, self.hidden_dim1)
        self.uc2 = torch.nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.uc3 = torch.nn.Linear(self.hidden_dim2, self.hidden_dim3)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):

        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_uncertain = e_discrimination * (stu_emb - k_difficulty)
        fully1 = self.drop(nn.Tanh()(self.fully1(input_x)))
        fully2 = self.drop(nn.Tanh()(self.fully2(fully1)))
        fully3 = self.drop(nn.Tanh()(self.fully3(fully2)))

        uc1 = self.drop(nn.Tanh()(self.fully1(input_uncertain)))
        uc2 = self.drop(nn.Tanh()(self.fully2(uc1)))
        uc3 = self.drop(nn.Tanh()(self.fully3(uc2)))

        gamma = torch.sigmoid(self.gamma(fully3))

        alpha = nn.Softplus()(self.alpha(uc3)) + 1  # + 1E-9
        beta = nn.Softplus()(self.beta(uc3))
        nu = nn.Softplus()(self.nu(uc3))  # + 1E-9

        return gamma, nu, alpha, beta

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data
    
    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data

