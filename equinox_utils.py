import io
import json
from jaxtyping import PyTree
import equinox

def debug_print(string, **kwargs):
    dynamic_kwargs = {k: equinox.partition(v, equinox.is_array)[0] for k, v in kwargs.items()}
    static_kwargs = {k: equinox.partition(v, equinox.is_array)[1] for k, v in kwargs.items()}
    equinox.tree_pprint(static_kwargs)
    jax.debug.print(string, **dynamic_kwargs)

def serialize(model):
    buffer = io.BytesIO()
    equinox.tree_serialise_leaves(buffer, model)
    return buffer.getvalue()

deserialize = lambda serialized_model, model: equinox.tree_deserialise_leaves(io.BytesIO(serialized_model), model)

def save(filename, hyperparams, model_bytes):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        f.write(model_bytes)

def load(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model_bytes = f.read()
        return hyperparams, model_bytes
    
def is_module_or_array(x):
    return isinstance(x, equinox.Module) or equinox.is_array(x) or callable(x)

def filter_tree_map(fn, tree, *rest):
    return jax.tree.map(
        fn,
        tree,
        *rest,
        is_leaf=is_module_or_array
    )
    
def scan(fn, y, xs):
    dynamic_y, static_y = equinox.partition(y, equinox.is_array)
    dynamic_xs, static_x = equinox.partition(xs, equinox.is_array)
    def body(dynamic_y, dynamic_x):
        y = equinox.combine(static_y, dynamic_y)
        x = equinox.combine(static_x, dynamic_x)
        y, z = fn(y, x)
        dynamic_y, _ = equinox.partition(y, equinox.is_array)
        return dynamic_y, z
    dynamic_y, z_list = jax.lax.scan(body, dynamic_y, dynamic_xs)
    y = equinox.combine(static_y, dynamic_y)
    return y, z_list

def module_stack(trees):
    dynamic_trees, static_trees = equinox.partition(trees, equinox.is_array)
    for tree in static_trees[1:]:
        if jax.tree.structure(tree) != jax.tree.structure(static_trees[0]):
            raise ValueError("All static trees must have the same structure")
    dynamic_trees = jax.tree.map(lambda *x: jax.numpy.stack(x), *dynamic_trees)
    return equinox.combine(static_trees[0], dynamic_trees)

class ModuleTree(equinox.Module):
    modules: PyTree[equinox.Module] = equinox.field(static=False)
    def __call__(self, x_tree, key=None):
        return filter_tree_map(lambda f, x: f(x, key=key), self.modules, x_tree)

import unittest
import jax

class TestEquinoxUtils(unittest.TestCase):
    def setUp(self):
        # Create a dummy model for testing
        self.dummy_model = equinox.nn.Linear(10, 5, key=jax.random.key(0))
        self.hyperparams = {"learning_rate": 0.001, "batch_size": 32}
        self.filename = "test_model.pkl"

    def test_serialize_deserialize(self):
        # Serialize the model
        serialized_model = serialize(self.dummy_model)
        
        # Deserialize the model
        deserialized_model = deserialize(serialized_model, self.dummy_model)
        
        # Check if the deserialized model is the same as the original
        self.assertTrue(equinox.tree_equal(self.dummy_model, deserialized_model))

    def test_save_load(self):
        # Serialize the model
        serialized_model = serialize(self.dummy_model)
        
        # Save the model and hyperparameters
        save(self.filename, self.hyperparams, serialized_model)
        
        # Load the model and hyperparameters
        loaded_hyperparams, loaded_model_bytes = load(self.filename)
        
        # Deserialize the loaded model
        loaded_model = deserialize(loaded_model_bytes, self.dummy_model)
        
        # Check if the loaded hyperparameters are the same as the original
        self.assertEqual(self.hyperparams, loaded_hyperparams)
        
        # Check if the loaded model is the same as the original
        self.assertTrue(equinox.tree_equal(self.dummy_model, loaded_model))

    def test_module_map(self):
        # Create a dictionary of MLPs
        key = jax.random.key(0)
        key1, key2 = jax.random.split(key)
        
        mlp_dict = {
            'feature1': equinox.nn.MLP(
                in_size=5,
                out_size=3,
                width_size=10,
                depth=2,
                activation=jax.nn.relu,
                key=key1
            ),
            'feature2': equinox.nn.MLP(
                in_size=7,
                out_size=3,
                width_size=12,
                depth=1,
                activation=jax.nn.relu,
                key=key2
            )
        }
        
        # Create input data with matching keys
        input_dict = {
            'feature1': jax.numpy.ones((5,)),
            'feature2': jax.numpy.ones((7,))
        }
        
        # Apply module_map to run each MLP on its corresponding input
        result = filter_tree_map(lambda module, xs: module(xs), mlp_dict, input_dict)
        
        # Check that the result has the expected structure and shapes
        self.assertIn('feature1', result)
        self.assertIn('feature2', result)
        self.assertEqual(result['feature1'].shape, (3,))
        self.assertEqual(result['feature2'].shape, (3,))

if __name__ == "__main__":
    unittest.main()
