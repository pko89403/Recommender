import timeit 

def timer(function):
    def new_function(*args, **kwargs):
        start_time = timeit.default_timer()
        res = function(*args, **kwargs)
        elapsed = timeit.default_timer() - start_time
        
        print(f"Function {function.__name__} took {elapsed} seconds to complete.")
        print(f"args - {args}")
        print(f"kwargs  - {kwargs}")
        return res 
    return new_function