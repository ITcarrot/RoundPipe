# Run Config

::: roundpipe.run_config.RoundPipeRunConfig.__init__

### split_input

It can take the following forms:

- `Tuple[Optional[Tuple], Optional[Dict[str, Any]]]`: A tuple of specs specifying positional and keyword arguments for splitting. The None value indicates automatic splitting (splitting tensor along the first dimension, replicate others).
- `Callable[[Tuple, Dict[str, Any], int], Tuple[List[Tuple], List[Dict[str, Any]]]]`: A custom function that takes the input arguments (positional and keyword), along with the number of microbatches, and returns a tuple containing lists of split positional and keyword arguments.
- `None`: Defaults to automatic splitting.

#### Writing Split Specs

[TODO]

#### Writing Custom Split Functions

[TODO]

### merge_output
It can take the following forms:

- `Any`: A spec defining how to merge the output.
- `Callable[[List[Any]], Any]`: A custom function that takes a list of output microbatches and merges them into a single output.
- `bool`: If True, uses default merging (concatenation for tensors along the first dimension, summation for numeric types). If False, returns some RoundPipePackedData objects.
- `None`: Defaults to True (default merging).

#### Writing Merge Specs

[TODO]

#### Writing Custom Merge Functions

[TODO]

#### No Merging Output

[TODO]
