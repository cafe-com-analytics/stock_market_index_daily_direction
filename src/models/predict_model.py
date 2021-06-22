from pgmpy.inference import VariableElimination

default_value = {'cluster_rt-1':  6.0, 'cluster_rt-5':  1.0, 'cluster_rt-37': 6.0}

def predict_model(model, evidence:dict=default_value):
    infer = VariableElimination(model)
    proba_predicted = infer.query(['cluster_rt'], evidence=evidence)
    valued_predicted = infer.map_query(['cluster_rt'], evidence=evidence)
    return (valued_predicted, proba_predicted)