import torch
import numpy as np

from .elliptical_slice import elliptical_slice, slice_sample
from .proj_model import ProjectedModel
from ..utils import sample_subspace_containing_vector, rejection_sample, hypersphere_surface_area, set_weights

class SubspaceGibbs(torch.nn.Module):
    def __init__(self, base, subspace, var, loader, criterion, num_samples = 20, 
                use_cuda = False, *args, **kwargs):
        super(SubspaceGibbs, self).__init__()

        self.base_model = base(*args, **kwargs)
        if use_cuda:
            self.base_model.cuda()

        self.base_params = []
        for name, param in self.base_model.named_parameters():
            self.base_params.append([param, name, param.size()])

        self.subspace = subspace
        self.var = var
        
        self.loader = loader
        self.criterion = criterion

        self.num_samples = num_samples
        self.use_cuda = use_cuda

        self.all_samples = None

        self.model = self.base_model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def prior_sample(self, prior='identity', scale=1.0):
        if prior=='identity':
            cov_mat = np.eye(self.subspace.cov_factor.size(0))

        elif prior=='schur':
            trans_cov_mat = self.subspace.cov_factor.matmul(self.subspace.cov_factor.subspace.t()).numpy()
            trans_cov_mat /= (self.swag_model.n_models.item() - 1)
            cov_mat = np.eye(self.subspace.cov_factor.size(0)) + trans_cov_mat

        else:
            raise NotImplementedError('Only schur and identity priors have been implemented')

        cov_mat *= scale
        sample = np.random.multivariate_normal(np.zeros(self.subspace.cov_factor.size(0)), cov_mat.astype(np.float64), 1)[0,:]
        return sample

    def log_pdf(self, params, temperature = 1., minibatch = False):
        params_tensor = torch.FloatTensor(params)
        params_tensor = params_tensor.view(-1)

        if self.use_cuda:
            params_tensor = params_tensor.cuda()
        with torch.no_grad():
            proj_model = ProjectedModel(model=self.base_model, subspace = self.subspace, proj_params = params_tensor)
            loss = 0
            num_datapoints = 0.0
            for batch_num, (data, target) in enumerate(self.loader):
                if minibatch and batch_num > 0:
                    break
                num_datapoints += data.size(0)
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                batch_loss, _, _ = self.criterion(proj_model, data, target)
                loss += batch_loss

        loss = loss / (batch_num+1) * num_datapoints
        return -loss.cpu().numpy() / temperature

    def fit(self, use_cuda = True, prior='identity', scale=1.0, **kwargs):
        # initialize at prior mean = 0
        current_sample = np.zeros(self.subspace.cov_factor.size(0))
        
        all_samples = np.zeros((current_sample.size, self.num_samples))
        logprobs = np.zeros(self.num_samples)
        for i in range(self.num_samples):
            prior_sample = self.prior_sample(prior=prior, scale=scale)
            current_sample, logprobs[i] = self.slice_method(initial_theta=current_sample, prior=prior_sample, 
                                                lnpdf=self.log_pdf,  **kwargs)
            # print(logprobs[i])
            all_samples[:,i] = current_sample
        
        self.all_samples = all_samples
        return logprobs

    def sample(self, ind=None, *args, **kwargs):
        if ind is None:
            ind = np.random.randint(self.num_samples)
        
        rsample = torch.FloatTensor(self.all_samples[:,int(ind)])

        #sample = self.subspace(torch.FloatTensor(rsample)).view(-1)

        if self.use_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        rsample = rsample.to(device)

        self.model = ProjectedModel(model=self.base_model, subspace=self.subspace, proj_params=rsample)
        return rsample


class SubspaceGibbs_:
    def __init__(self, data, model, center, W_init, eps=0.05):  # priors?? How to sample theta in posterior?
        """
        Generative Model:
          center (fixed)
          W |center ~ uniform over directions
          theta | W, center = phi * s, where
                phi | W, center ~ uniform over directions within W
                s ~ chi squared(d-1) or some other distro over magnitudes
          y | theta, x = NN(x; theta) + eps(x; theta)


        :param data: (`torch.tensor`) n x D tensor of data
        :param model: (`torch.nn.Module`) the probabilistic model
        :param center: (`torch.tensor`) the SWA solution (d-dimensional flattened parameter vector)
        :param W_init: (`torch.tensor`) an initial subspace basis. Should be k x d dimensional with orthonormal rows.
        :param eps: (`float`) the amount that we scale the SWA likelihood by to ensure rejection sampling is kosher.
        """
        self.k, self.d = W_init.shape
        self.center = center
        self.subspace_offset = torch.zeros(self.k)
        self.subspace = W_init
        self.data = data
        self.model = model

        self.magnitude_prior = torch.distributions.Chi2(self.k)  # TODO: make this optional/modifiable.
        self.eps = eps
        self._swa_log_lik = None

    def log_likelihood(self, model, loader, minibatch=False):
        # if self.use_cuda:
        #     params_tensor = params_tensor.cuda()
        with torch.no_grad():
            # proj_model = ProjectedModel(model=self.base_model, subspace = self.subspace, proj_params = params_tensor)
            loss = 0
            num_datapoints = 0.0
            batch_num = -1
            for batch_num, (data, target) in enumerate(loader):
                if minibatch and batch_num > 0:
                    break
                num_datapoints += data.size(0)
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()
                pred = model(data)
                batch_loss = (pred[:, 0] - target).pow(2) / pred[:, 1]  # TODO: expand to other kinds of likelihoods.
                loss += batch_loss

        loss = loss / (batch_num+1) * num_datapoints
        return -loss.cpu().numpy()

    @property
    def offset(self):
        return self.subspace_offset @ self.subspace  # probably not needed. TODO: Refactor.

    @property
    def theta(self):
        return self.center + self.offset

    def log_subspace_offset_conditional_prior(self, z):
        """P(z | W, theta_0)"""
        mag_p = self.magnitude_prior.log_prob(z.pow(2).sum())
        dir_p = -torch.log(hypersphere_surface_area(self.d))
        return mag_p + dir_p

    def sample_subspace_offset(self):
        direction = torch.randn(self.k)
        direction = direction / torch.norm(direction)

        magnitude = self.magnitude_prior.sample()
        return direction * magnitude.pow(1/2)

    def log_subspace_offset_prior_lik(self, z):
        theta = z @ self.subspace + self.center
        set_weights(self.model, theta)
        return self.log_likelihood(self.model, self.data) + self.log_subspace_offset_conditional_prior(z)

    def log_scaled_proposal_prob(self, z):
        return self.swa_log_lik + torch.log(1 + self.eps) + self.log_subspace_offset_conditional_prior(z)

    @property
    def swa_log_lik(self):
        if self._swa_log_lik is None:
            set_weights(self.model, self.center)
            self._swa_log_lik = self.log_likelihood(self.model, self.data)
        return self._swa_log_lik

    def sample(self, n_samples=1):
        samples = []
        for i in range(n_samples):
            # Sample subspace
            # P(W | theta, D) = P(D | theta, W) P(W | theta) / Z
            #                 = P(D | theta) P(theta | W) P(W) / Z2
            #                 = P(D | theta) {1 if theta in W} / Z3
            #                 = {1 if theta in W} / Z4
            self.subspace = sample_subspace_containing_vector(self.offset, self.k)

            # maybe could use adaptive rejection sampling?
            self.subspace_offset = rejection_sample(self.log_subspace_offset_prior_lik,
                                                    self.log_scaled_proposal_prob,
                                                    self.sample_subspace_offset, using_logs=True).send(None)
            samples.append(self.theta)
        return samples
