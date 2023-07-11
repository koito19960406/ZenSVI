# Copyright (c) Facebook, Inc. and its affiliates. (http://www.facebook.com)
# -*- coding: utf-8 -*-

"""
mapillary.models.exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains the set of Mapillary Exceptions used internally.

For more information, please check out https://www.mapillary.com/developer/api-documentation/.

- Copyright: (c) 2021 Facebook
- License: MIT LICENSE
"""

# Package imports
import typing


class MapillaryException(Exception):
    """Base class for exceptions in this module"""

    pass


class InvalidBBoxError(MapillaryException):
    """
    Raised when an invalid coordinates for bounding box are entered
    to access Mapillary's API.

    :var message: The error message returned
    :type message: str
    """

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return f'InvalidBBoxError: "{self.message}" '

    def __repr__(self):
        return f'InvalidBBoxError: = "{self.message}" '


class InvalidTokenError(MapillaryException):
    """
    Raised when an invalid token is given
    to access Mapillary's API, primarily used in mapillary.set_access_token

    :var message: The error message returned
    :type message: str

    :var error_type: The type of error that occurred
    :type error_type: str

    :var code: The error code returned, most likely 190, "Access token has expired".
        See https://developers.facebook.com/docs/graph-api/using-graph-api/error-handling/
        for more information
    :type code: str

    :var fbtrace_id: A unique ID to track the issue/exception
    :type fbtrace_id: str
    """

    def __init__(self, message: str, error_type: str, code: str, fbtrace_id: str):
        """
        Initializing InvalidTokenError constructor

        :param message: Error message
        :type message: str

        :param error_type: Type of error
        :type error_type: str

        :param code: The error code
        :type code: str

        :param fbtrace_id: the FBTrace_ID
        :type fbtrace_id: str
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
    """
    Raised when a function is called without having the access token set in
    set_access_token to access Mapillary's API, primarily used in mapillary.set_access_token

    :var message: The error message returned
    :type message: str
    """

    def __init__(self, message: str):
        """
        Initializing AuthError constructor

        :param message: Error message
        :type message: str
        """

        self.message = message

    def __str__(self):
        return f'AuthError: An exception occurred, "{self.message}"'

    def __repr__(self):
        return "AuthError: An exception occurred." + f'Message: "{self.message}"'


class InvalidImageResolutionError(MapillaryException):
    """
    Raised when trying to retrieve an image thumbnail with an invalid resolution/size.

    Primarily used with mapillary.image_thumbnail

    :var resolution: Image size entered by the user
    :type resolution: int
    """

    def __init__(self, resolution: int) -> None:
        """
        Initialize InvalidImageResolutionError constructor

        :param resolution: Image resolution
        :type resolution: int
        """

        self._resolution = resolution

    def __str__(self) -> str:
        return f"""An exception occurred, "{self._resolution}" is not a supported image size

Hint: Supported image sizes are: 256, 1024, and 2048
        """

    def __repr__(self) -> str:
        return (
            f'An exception occurred, "{self._resolution}" is not a supported image size'
        )


class InvalidImageKeyError(MapillaryException):
    """
    Raised when trying to retrieve an image thumbnail with an invalid image ID/key.
    Primarily used with mapillary.image_thumbnail

    :var image_id: Image ID/key entered by the user
    :param image_id: int
    """

    def __init__(self, image_id: typing.Union[int, str]) -> None:
        """
        Initializing InvalidImageKeyError constructor

        :param image_id: The image id
        :type image_id: int|str
        """

        self._image_id = image_id

    def __str__(self) -> str:
        return f'An exception occurred, "{self._image_id}" is not a valid image ID/key'

    def __repr__(self) -> str:
        return f'An exception occurred, "{self._image_id}" is not a valid image ID/key'


class InvalidKwargError(MapillaryException):
    """
    Raised when a function is called with the invalid keyword argument(s) that do not belong to the
    requested API end call

    :var func: The function that was called
    :type func: str

    :var key: The key that was passed
    :type key: str

    :var value: The value along with that key
    :type value: str

    :var options: List of possible keys that can be passed
    :type options: list
    """

    def __init__(
        self,
        func: str,
        key: str,
        value: str,
        options: list,
    ):
        """
        Initializing InvalidKwargError constructor

        :param func: The function that was called
        :type func: str

        :param key: The key that was passed
        :type key: str

        :param value: The value along with that key
        :type value: str

        :param options: List of possible keys that can be passed
        :type options: list
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
    """
    Out of bound zoom error

    :var param: The invalid param passed
    :type param: str

    :var value: The invalid value passed
    :type value: any

    :var options: The possible list of zoom values
    :type options: list
    """

    def __init__(
        self,
        param: str,
        value: any,
        options: list,
    ):
        """
        Initializing InvalidOptionError constructor

        :param param: The invalid param passed
        :type param: str

        :param value: The invalid value passed
        :type value: any

        :param options: The possible list of zoom values
        :type options: list
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
    """
    Raised when an API endpoint is passed invalid field elements

    :var endpoint: The API endpoint that was targeted
    :type endpoint: str

    :var field: The invalid field that was passed
    :type field: list
    """

    def __init__(
        self,
        endpoint: str,
        field: list,
    ):
        """
        Initializing InvalidFieldError constructor

        :param endpoint: The API endpoint that was targeted
        :type endpoint: str

        :param field: The invalid field that was passed
        :type field: list
        """

        self.endpoint = endpoint
        self.field = field

    def __str__(self):
        return (
            f'InvalidFieldError: The invalid field, "{self.field}" was '
            f'passed to the endpoint, "{self.endpoint}"'
        )

    def __repr__(self):
        return (
            f'InvalidFieldError: The invalid field, "{self.field}" was '
            f'passed to the endpoint, "{self.endpoint}"'
        )


class LiteralEnforcementException(MapillaryException):
    """
    Raised when literals passed do not correspond to options
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

    @staticmethod
    def enforce_literal(
        option_selected: str,
        options: typing.Union[typing.List[str], typing.List[int]],
        param: str,
    ):

        if option_selected not in options:

            raise InvalidOptionError(
                param=param, value=option_selected, options=options
            )


class InvalidNumberOfArguments(MapillaryException):
    """
    Raised when an inappropriate number of parameters are passed to a function
    """

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
