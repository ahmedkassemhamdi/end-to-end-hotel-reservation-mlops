import traceback
import inspect


class CustomException(Exception):
    
    def __init__(self, message: str, include_traceback: bool = True):
        self.message = message
        if include_traceback:
            self.message = self._add_traceback_info(message)
        super().__init__(self.message)
    
    def _add_traceback_info(self, message: str) -> str:

        try:
            stack = inspect.stack()
            frame = stack[2]
            filename = frame.filename
            lineno = frame.lineno
            function = frame.function
            
            return f"Error in {filename}, line {lineno}, function {function}: {message}"
        except:
            return message  
    
    def __str__(self):
        return self.message
