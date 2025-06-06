import einops
import jax

def random_tree(tree_definition, key):
    keys = jax.random.split(key, tree_definition.num_leaves)
    return jax.tree.unflatten(tree_definition, keys)

def random_tree_like(tree, key):
    return jax.tree.map(lambda leaf, key: jax.random.normal(key, leaf.shape), tree, random_tree(jax.tree.structure(tree), key))

def tree_stack(trees, axis):
    return jax.tree.map(lambda *arrays: jax.numpy.stack(arrays, axis=axis), *trees)

def tree_unstack(trees, axis):
    nested_trees = jax.tree.map(lambda x: jax.numpy.unstack(x, axis=axis), trees)
    return jax.tree.transpose(jax.tree.structure(trees), None, nested_trees)

def tree_pack(data_trees, pattern_tree):
    data_tree = jax.tree.map(lambda pattern, *data: einops.pack(data, pattern)[0], pattern_tree, *data_trees)
    shape_tree = jax.tree.map(lambda pattern, *data: einops.pack(data, pattern)[1], pattern_tree, *data_trees)
    return data_tree, shape_tree

def tree_unpack(data_tree, shape_tree, pattern_tree):
    nested_trees = jax.tree.map(lambda pattern, data, shape: einops.unpack(data, shape, pattern), pattern_tree, data_tree, shape_tree)
    return jax.tree.transpose(jax.tree.structure(pattern_tree), None, nested_trees)

def tree_concat(trees, axis):
    return jax.tree.map(lambda *arrays: jax.numpy.concatenate(arrays, axis=axis), *trees)

def tree_split(trees, indices_or_sections, axis):
    nested_trees = jax.tree.map(lambda x: jax.numpy.split(x, indices_or_sections, axis=axis), trees)
    return jax.tree.transpose(jax.tree.structure(trees), None, nested_trees)

def vmap_like(x, value):
    return jax.vmap(lambda _: value)(x)

lax_for_in_loop = lambda f, sequence: jax.lax.fori_loop(0, len(sequence), lambda index: f(sequence[index]))

import equinox
def test_stack_unstack():
    """
    Test the tree_stack and tree_unstack functions to ensure they work correctly.
    
    This function creates a simple tree structure, stacks multiple instances,
    and then unstacks them to verify the original trees are recovered.
    It also tests different axis values for stacking and unstacking.
    
    Returns:
        bool: True if the test passes, False otherwise
    """
    # Create some simple test trees
    tree1 = {"a": jax.numpy.array([1, 2, 3]), "b": {"c": jax.numpy.array([4, 5, 6])}}
    tree2 = {"a": jax.numpy.array([7, 8, 9]), "b": {"c": jax.numpy.array([10, 11, 12])}}
    tree3 = {"a": jax.numpy.array([13, 14, 15]), "b": {"c": jax.numpy.array([16, 17, 18])}}
    
    # Test default axis (0)
    stacked = tree_stack([tree1, tree2, tree3])
    unstacked = tree_unstack(stacked)
    
    equinox.tree_pprint(stacked)
    equinox.tree_pprint(unstacked)
    
    # Check if the unstacked trees match the original trees
    all_match = True
    for original, recovered in zip([tree1, tree2, tree3], unstacked):
        for key in original:
            if isinstance(original[key], dict):
                for subkey in original[key]:
                    if not jax.numpy.array_equal(original[key][subkey], recovered[key][subkey]):
                        all_match = False
            else:
                if not jax.numpy.array_equal(original[key], recovered[key]):
                    all_match = False
    
    # Test with axis=1
    # Create trees with arrays that can be stacked along axis 1
    tree1_axis = {"a": jax.numpy.array([[1, 2], [3, 4]]), "b": {"c": jax.numpy.array([[5, 6], [7, 8]])}}
    tree2_axis = {"a": jax.numpy.array([[9, 10], [11, 12]]), "b": {"c": jax.numpy.array([[13, 14], [15, 16]])}}
    
    # Stack along axis 1
    stacked_axis1 = tree_stack([tree1_axis, tree2_axis], axis=1)
    # Expected shapes: stacked_axis1.a: (2, 2, 2), stacked_axis1.b.c: (2, 2, 2)
    
    # Unstack along axis 1
    unstacked_axis1 = tree_unstack(stacked_axis1, axis=1)
    
    equinox.tree_pprint(stacked_axis1)
    equinox.tree_pprint(unstacked_axis1)
    
    # Check if the unstacked trees match the original trees
    for original, recovered in zip([tree1_axis, tree2_axis], unstacked_axis1):
        for key in original:
            if isinstance(original[key], dict):
                for subkey in original[key]:
                    if not jax.numpy.array_equal(original[key][subkey], recovered[key][subkey]):
                        all_match = False
            else:
                if not jax.numpy.array_equal(original[key], recovered[key]):
                    all_match = False
    
    return all_match
# print(test_stack_unstack())

def test_concat_split():
    """Test the tree_concat and tree_split functions."""
    # Create test trees
    tree1 = {"a": jax.numpy.array([1, 2, 3]), "b": {"c": jax.numpy.array([4, 5, 6])}}
    tree2 = {"a": jax.numpy.array([7, 8, 9]), "b": {"c": jax.numpy.array([10, 11, 12])}}
    tree3 = {"a": jax.numpy.array([13, 14, 15]), "b": {"c": jax.numpy.array([16, 17, 18])}}
    
    # Test concatenation along axis 0
    concatenated = tree_concat([tree1, tree2, tree3])
    
    # Test splitting along axis 0
    # We can specify the number of sections or the indices at which to split
    split_trees = tree_split(concatenated, 3)  # Split into 3 equal sections
    
    # Also test with specific indices
    split_trees_indices = tree_split(concatenated, [3, 6])  # Split at indices 3 and 6
    
    equinox.tree_pprint(concatenated)
    equinox.tree_pprint(split_trees)
    equinox.tree_pprint(split_trees_indices)
    
    # Check if the split trees match the original trees
    all_match = True
    for original, split in zip([tree1, tree2, tree3], split_trees):
        for key in original:
            if isinstance(original[key], dict):
                for subkey in original[key]:
                    if not jax.numpy.array_equal(original[key][subkey], split[key][subkey]):
                        all_match = False
            else:
                if not jax.numpy.array_equal(original[key], split[key]):
                    all_match = False
    
    # Test with different axis
    # Create trees with arrays that can be concatenated along axis 1
    tree1_axis = {"a": jax.numpy.array([[1, 2], [3, 4]]), "b": {"c": jax.numpy.array([[5, 6], [7, 8]])}}
    tree2_axis = {"a": jax.numpy.array([[9, 10], [11, 12]]), "b": {"c": jax.numpy.array([[13, 14], [15, 16]])}}
    
    # Concatenate along axis 1
    concatenated_axis1 = tree_concat([tree1_axis, tree2_axis], axis=1)
    
    # Split along axis 1
    # We can use either the number of sections or specific indices
    split_trees_axis1 = tree_split(concatenated_axis1, 2, axis=1)  # Split into 2 equal sections
    split_trees_axis1_indices = tree_split(concatenated_axis1, [2], axis=1)  # Split at index 2
    
    equinox.tree_pprint(concatenated_axis1)
    equinox.tree_pprint(split_trees_axis1)
    equinox.tree_pprint(split_trees_axis1_indices)
    
    # Check if the split trees match the original trees
    axis1_match = True
    for original, split in zip([tree1_axis, tree2_axis], split_trees_axis1):
        for key in original:
            if isinstance(original[key], dict):
                for subkey in original[key]:
                    if not jax.numpy.array_equal(original[key][subkey], split[key][subkey]):
                        axis1_match = False
            else:
                if not jax.numpy.array_equal(original[key], split[key]):
                    axis1_match = False
    
    return all_match and axis1_match
# print(test_concat_split())
