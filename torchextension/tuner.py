# === Glory Be To God ====

# import library
import torch
from typing import List, Dict, Optional, Union
from torchmodel import Sequential


class Hyperparameter(object):
    """
    The class contains methods for setting
    the model parameter
    """

    def __init__(self):
        """
        Class Constructor for initializing
        class parameter

        **Return:**
            None
        """
        self.param = dict()
        self.select = {}

    def Choice(self, name, options) -> any:
        """
        The method is used to set name of    
        parameter with its options pack the value
        from

        **Parameter:**
            name(str): Name of the parameter
            options(list): List of options

        **Return:**
            Any value from options
        """

        # import hidden torch libraries
        import torch

        # Set the name as key and options as values to param parameter of the class
        self.param[name] = torch.tensor(options)

        # Check if the name is not in select dictionary then set the name as key and give it
        # a default value of 0
        if name not in self.select:
            self.select[name] = 0
        else:
            pass

        try:
            return options[self.select[name]]
        except IndexError:
            # if options receives a select value which is out range then assign it to 0
            self.select[name] = 0
            return options[self.select[name]]

    def Int(self, name: str, min_value: int, max_value: int, step: int = None) -> int:
        """
        This method sets the name of the 
        parameter with  values from min_value to 
        max_value and jump each step if the step
        parameter is not none.

        **Parameters:**
            name (str): Name of the parameter.
            min_value(int): The value to start from.
            max_value(int):  Value to end with.
            step (int): Range of value to jump.

        **Return:**
            Integer from the range.
        """
        # import hidden torch libraries
        import torch

        # Construct options
        if step:
            options = torch.tensor(range(min_value, max_value + min_value, step))
        else:
            options = torch.tensor(range(min_value, max_value + 1, 1))

        # Set the name as key and options as values to param parameter of the class
        self.param[name] = options

        # Check if the name is not in select dictionary then set the name as key and give it
        # a default value of 0
        if name not in self.select:
            self.select[name] = 0
        else:
            pass

        try:
            return options[self.select[name]]
        except IndexError:
            # if options receives a select value which is out range then assign it to 0
            self.select[name] = 0
            return options[self.select[name]]

    def Float(self, name: str, min_value: any, max_value: any) -> int:
        """
        This method sets the name of the
        parameter with  values from min_value to
        max_value and jump each step if the step
        parameter is not none

        **parameters:**
            name (str): Name of the parameter.
            min_value(float): The value to start from.
            max_value(float):  Value to end with.

        **Return**
            : Float in the range min_values and max_value.
        """
        # import hidden torch libraries
        import numpy as np

        # Construct options
        options = np.linspace(min_value, max_value)

        # Set the name as key and options as values to param parameter of the class
        self.param[name] = options

        # Check if the name is not in select dictionary then set the name as key and give it
        # a default value of 0
        if name not in self.select:
            self.select[name] = 0
        else:
            pass

        try:
            return options[self.select[name]]
        except IndexError:
            # if options receives a select value which is out of range then assign it to 0
            self.select[name] = 0
            return options[self.select[name]]

    def selector(self, name) -> None:
        """
        The method adds 1 on the selected key value
        # Parameters:
            name (str): The name used as key in select dictionary to add 1 on its value
        # Return:
            NoneType
        """
        if self.select[name] < len(self.param[name]):
            self.select[name] += 1

        return self.select[name]


class Tuner(object):
    """
    The class contains methods for running 
    __model while tuning its parameters
    """

    def __init__(self, model: any, n_trials: int, objective: str = None) -> None:
        self.__model: any = model
        self.__trials: int = n_trials
        self.__objective: str = objective
        self.__MODELS = {}

    def __best_models(self, n_models: Union[int, str] = 1) -> List[Dict]:

        # Initialize the models.
        models = self.__MODELS

        # Make dictionary with __model name as values and __objective as key.
        objectives_scored = dict()

        # loop over the models.
        for key, value in models.items():
            objective_score = self.__MODELS[key]['history'][self.__objective][-1]
            objectives_scored[objective_score] = key

        if n_models == "all":
            return [models[list(objectives_scored.values())[i]] for i in range(len(objectives_scored))]
        return [models[list(objectives_scored.values())[i]] for i in range(n_models)]

    def get_best_models(self, n_models: Union[int, str] = 1) -> List[Sequential]:
        """
        Returns the list of best model.

        **Parameters:**
            n_models: (int | str), number of models to return. default (1)
                    Use 'all' to return all the models in the list.
                    use integer to return some few models.
        **return:**
            list of best models.
        """
        return [model['__model'] for model in self.__best_models(n_models)]

    @property
    def best_model(self) -> Sequential:
        """
        Return the best model from all the model.

        **return:**
            Sequential.
        """
        return self.get_best_models()[0]

    def search(self, x, epoch: int = 1, random_state: Optional[int] = 0):

        # Set the random_state for reproducibility.
        torch.manual_seed(random_state)

        # Instantiate Hyperparameter object.
        hyperparameter = Hyperparameter()

        # Instantiate a variable to track the best score.
        best_score = 0

        for i in range(self.__trials):

            print()
            print("\n\033[1mNumber of trials: %d\033[0m" % i)

            # Initialize the __model function.
            model = self.__model(hyperparameter)

            # Fit __model.
            history = model.fit(x, epochs=epoch)

            # Append every __model with its history.
            self.__MODELS.update({f'model_{i}': {'__model': model, 'history': history}})

            if self.__objective:
                score = self.__MODELS['model_' + str(i)]['history'][self.__objective][-1]

                if best_score == 0:
                    best_score = score

                if score < best_score:
                    best_score = score

                print('\n')
                print(f'|\033[1mBest score from {self.__objective}: {best_score}\033[0m')

            print()
            print("\n\033[1mModel Parameters:\033[0m")
            print()
            for child in model.children():
                for c in child.children():
                    for m in c.children():
                        model_param = "| " + str(m)
                        new_model_param = (model_param
                                           .replace("(", " - ")
                                           .replace(")", " ")
                                           .replace(",", " - ")
                                           )
                        print(new_model_param)
                # print(__model.optim.param_groups)

            for key, value in hyperparameter.param.items():

                # Print the optim summary
                try:
                    print(f"\n| \033[1m{key}\033[0m - {model.optim.param_groups[0][key]}")
                except KeyError:
                    pass
                hyperparameter.selector(key)
                # print(hyperparameter.selector(key))
            print('=' * 60)

#                                   Glory Be to God
