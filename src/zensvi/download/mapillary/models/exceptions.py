# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-
"""mapillary.models.exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the set of Mapillary Exceptions used internally.

For more information, please check out https://www.mapillary.com/developer/api-documentation/.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

# Package imports
import typing


class MapillaryException(Exception):
    """Base class for exceptions in this module."""

    pass


class InvalidBBoxError(MapillaryException):
    """Raised when an invalid coordinates for bounding box are entered to access
    Mapillary's API.

    Attributes:
        message (str): The error message returned
    """

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return f'InvalidBBoxError: "{self.message}" '

    def __repr__(self):
        return f'InvalidBBoxError: = "{self.message}" '


class InvalidTokenError(MapillaryException):
    """Raised when an invalid token is given to access Mapillary's API, primarily used
    in mapillary.set_access_token.

    Attributes:
        message (str  :var error_type: The type of error that occurred):
            The error message returned :type message: str
        error_type (str :var code: The error code returned, most likely 190, "Access token has expired". See https://developers.facebook.com/docs/graph-api/using- graph-api/error-handling/ for more information):
            The type of error that occurred :type error_type: str
        code (str  :var fbtrace_id: A unique ID to track the issue/exception):
            The error code returned, most likely 190, "Access token has
            expired".

    See
    https://developers.facebook.com/docs/graph-api/using-graph-api/error-handling/

    Args:
        fbtrace_id (str)
    """

    def __init__(self, message: str, error_type: str, code: str, fbtrace_id: str):
        """Initializing InvalidTokenError constructor.

        Args:
            message (str): Error message
            error_type (str): Type of error
            code (str): The error code
            fbtrace_id (str): the FBTrace_ID
        """
        self.message = message
        self.error_type = error_type
        self.code = code
        self.fbtrace_id = fbtrace_id

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return (
            "InvalidTokenError: An exception occurred."
            + f'Message: "{self.message}", Type: "{self.error_type}",'
            + f'Code: "{self.code}",'
            + f'fbtrace_id: "{self.fbtrace_id}"'
        )


class AuthError(MapillaryException):
    """Raised when a function is called without having the access token set in
    set_access_token to access Mapillary's API, primarily used in
    mapillary.set_access_token.

    Attributes:
        message (str): The error message returned
    """

    def __init__(self, message: str):
        """Initializing AuthError constructor.

        Args:
            message (str): Error message
        """
        self.message = message

    def __str__(self):
        return f'AuthError: An exception occurred, "{self.message}"'

    def __repr__(self):
        return "AuthError: An exception occurred." + f'Message: "{self.message}"'


class InvalidImageResolutionError(MapillaryException):
    """Raised when trying to retrieve an image thumbnail with an invalid
    resolution/size.

    Primarily used with mapillary.image_thumbnail

    Attributes:
        resolution (int): Image size entered by the user
    """

    def __init__(self, resolution: int) -> None:
        """Initialize InvalidImageResolutionError constructor.

        Args:
            resolution (int): Image resolution
        """
        self._resolution = resolution

    def __str__(self) -> str:
        return f"""An exception occurred, "{self._resolution}" is not a supported image size

Hint: Supported image sizes are: 256, 1024, and 2048
        """

    def __repr__(self) -> str:
        return f'An exception occurred, "{self._resolution}" is not a supported image size'


class InvalidImageKeyError(MapillaryException):
    """Raised when trying to retrieve an image thumbnail with an invalid image ID/key.
    Primarily used with mapillary.image_thumbnail.

    Attributes:
        image_id: Image ID/key entered by the user

    Args:
        image_id: int
    """

    def __init__(self, image_id: typing.Union[int, str]) -> None:
        """Initializing InvalidImageKeyError constructor.

        Args:
            image_id (int|str): The image id
        """
        self._image_id = image_id

    def __str__(self) -> str:
        return f'An exception occurred, "{self._image_id}" is not a valid image ID/key'

    def __repr__(self) -> str:
        return f'An exception occurred, "{self._image_id}" is not a valid image ID/key'


class InvalidKwargError(MapillaryException):
    """Raised when a function is called with the invalid keyword argument(s) that do not
    belong to the requested API end call.

    Attributes:
        func (str  :var key: The key that was passed): The function that
            was called

    Args:
        key (str  :var value: The value along with that key)
        value (str  :var options: List of possible keys that can be passed)
        options (list)
    """

    def __init__(
        self,
        func: str,
        key: str,
        value: str,
        options: list,
    ):
        """Initializing InvalidKwargError constructor.

        Args:
            func (str): The function that was called
            key (str): The key that was passed
            value (str): The value along with that key
            options (list): List of possible keys that can be passed
        """
        self.func = func
        self.key = key
        self.value = value
        self.options = options

    def __str__(self):
        return (
            f'InvalidKwargError: The invalid kwarg, ["{self.key}": '
            f'{self.value}] was passed to the function, "{self.func}".\n'
            f"A possible list of keys for this function are, "
            f'{", ".join(self.options)}'
        )

    def __repr__(self):
        return (
            f'InvalidKwargError: The invalid kwarg, ["{self.key}": '
            f'{self.value}] was passed to the function, "{self.func}".\n'
            f"A possible list of keys for this function are, "
            f'{", ".join(self.options)}'
        )


class InvalidOptionError(MapillaryException):
    """Out of bound zoom error.

    Attributes:
        param (str  :var value: The invalid value passed): The invalid
            param passed

    Args:
        value (any  :var options: The possible list of zoom values)
        options (list)
    """

    def __init__(
        self,
        param: str,
        value: any,
        options: list,
    ):
        """Initializing InvalidOptionError constructor.

        Args:
            param (str): The invalid param passed
            value (any): The invalid value passed
            options (list): The possible list of zoom values
        """
        self.param = param
        self.value = value
        self.options = options

    def __str__(self):
        return (
            f'InvalidOptionError: Given {self.param} value, "{self.value}" '
            f'while possible {self.param} options, [{", ".join(self.options)}] '
        )

    def __repr__(self):
        return (
            f'InvalidOptionError: Given {self.param} value, "{self.value}" '
            f'while possible {self.param} options, [{", ".join(self.options)}] '
        )


class InvalidFieldError(MapillaryException):
    """Raised when an API endpoint is passed invalid field elements.

    Attributes:
        endpoint (str  :var field: The invalid field that was passed):
            The API endpoint that was targeted

    Args:
        field (list)
    """

    def __init__(
        self,
        endpoint: str,
        field: list,
    ):
        """Initializing InvalidFieldError constructor.

        Args:
            endpoint (str): The API endpoint that was targeted
            field (list): The invalid field that was passed
        """
        self.endpoint = endpoint
        self.field = field

    def __str__(self):
        return f'InvalidFieldError: The invalid field, "{self.field}" was ' f'passed to the endpoint, "{self.endpoint}"'

    def __repr__(self):
        return f'InvalidFieldError: The invalid field, "{self.field}" was ' f'passed to the endpoint, "{self.endpoint}"'


class LiteralEnforcementException(MapillaryException):
    """Raised when literals passed do not correspond to options."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

    @staticmethod
    def enforce_literal(
        option_selected: str,
        options: typing.Union[typing.List[str], typing.List[int]],
        param: str,
    ):
        """Enforce that a selected option is one of the allowed options.

        Args:
            option_selected (str): The option that was selected
            options (Union[List[str], List[int]]): List of valid options to choose from
            param (str): Name of the parameter being validated

        Raises:
            InvalidOptionError: If option_selected is not in options
        """
        if option_selected not in options:
            raise InvalidOptionError(param=param, value=option_selected, options=options)


class InvalidNumberOfArguments(MapillaryException):
    """Raised when an inappropriate number of parameters are passed to a function."""

    def __init__(
        self,
        number_of_params_passed: int,
        actual_allowed_params: int,
        param: str,
        *args: object,
    ) -> None:
        super().__init__(*args)

        self.number_of_params_passed = number_of_params_passed
        self.actual_allowed_params = actual_allowed_params
        self.param = param

    def __str__(self):
        return (
            f'InvalidNumberOfArguments: The parameter, "{self.param}" was '
            f'passed "{self.number_of_params_passed}" items, when the max number of'
            f'allowed items are "{self.actual_allowed_params}"'
        )

    def __repr__(self):
        return (
            f"InvalidNumberOfArguments(number_of_params_passed={self.number_of_params_passed},"
            f"actual_allowed_params={self.actual_allowed_params}, param={self.param})"
        )
