import yaml

def load_inputs(path:str="inputs.yaml") -> dict:
    """
    Load inputs from a YAML file.

    Args:
        path (str): The path to the YAML file. Defaults to "inputs.yaml".

    Returns:
        dict: A dictionary containing the inputs loaded from the YAML file.
    """

    if not path.endswith(".yaml"):
        path += ".yaml"

    with open(path, "r") as file:
        inputs = yaml.safe_load(file)
    return inputs