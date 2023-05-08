#%%
from numba import jit, int32, float64
from numba.experimental import jitclass
from numba.types import Array
import numpy as np

# Define a structure to hold the data for an object
object_spec = [
    ('id', int32),
    ('x', float64),
    ('y', float64),
]

# Define the JIT class specification
spec = [
    ('my_array', Array())
]

# Define the JIT class
@jitclass(spec)
class ObjectProcessor:
    def __init__(self, my_array):
        self.my_array = my_array
    
    def process_objects(self):
        result = []
        for obj in self.my_array:
            result.append(obj.x + obj.y)
        return result

# Create an instance of the ObjectProcessor class
my_array = np.array([(1, 1.0, 2.0), (2, 3.0, 4.0), (3, 5.0, 6.0)], dtype=object_spec)
processor = ObjectProcessor(my_array)

# Call the process_objects method
result = processor.process_objects()
print(result)
# %%
