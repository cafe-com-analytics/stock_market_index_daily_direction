from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel

default_list = [
    ('cluster_rt-37', 'cluster_rt'),
    ('cluster_rt-5', 'cluster_rt'),
    ('cluster_rt-1', 'cluster_rt')]

def train_model(df, lst_relations: list=default_list) -> BayesianModel:
    model = BayesianModel(lst_relations)

    # model.cpds = []
    model.fit(df, 
            estimator=BayesianEstimator,
            prior_type="k2",
            equivalent_sample_size=10,
            complete_samples_only=False)
    
    return model