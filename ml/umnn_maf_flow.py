from UMNN import UMNNMAFFlow


class UmnnMafFlow(UMNNMAFFlow):
    def forward_and_compute_log_jacobian(self, y, context=None):
        """
        API needed in inference.neb.elbo.py & inference.neb.iw.py
        """
        z, log_jacobian = self.compute_log_jac_bis(y, context)
        return z, log_jacobian.sum(1)
