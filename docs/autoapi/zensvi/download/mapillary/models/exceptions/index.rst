:py:mod:`zensvi.download.mapillary.models.exceptions`
=====================================================

.. py:module:: zensvi.download.mapillary.models.exceptions

.. autoapi-nested-parse::

   mapillary.models.exceptions
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This module contains the set of Mapillary Exceptions used internally.

   For more information, please check out https://www.mapillary.com/developer/api-documentation/.

   - Copyright: (c) 2021 Facebook
   - License: MIT LICENSE



Module Contents
---------------

.. py:exception:: MapillaryException


   Bases: :py:obj:`Exception`

   Base class for exceptions in this module

   .. py:class:: __cause__


      exception cause


   .. py:class:: __context__


      exception context


   .. py:class:: __suppress_context__



   .. py:class:: __traceback__



   .. py:class:: args



   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __setattr__()

      Implement setattr(self, name, value).


   .. py:method:: __setstate__()


   .. py:method:: __sizeof__()

      Size of object in memory, in bytes.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



.. py:exception:: InvalidBBoxError(message: str)


   Bases: :py:obj:`MapillaryException`

   Raised when an invalid coordinates for bounding box are entered
   to access Mapillary's API.

   :var message: The error message returned
   :type message: str

   .. py:class:: __cause__


      exception cause


   .. py:class:: __context__


      exception context


   .. py:class:: __suppress_context__



   .. py:class:: __traceback__



   .. py:class:: args



   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __setattr__()

      Implement setattr(self, name, value).


   .. py:method:: __setstate__()


   .. py:method:: __sizeof__()

      Size of object in memory, in bytes.


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



.. py:exception:: InvalidTokenError(message: str, error_type: str, code: str, fbtrace_id: str)


   Bases: :py:obj:`MapillaryException`

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

   .. py:class:: __cause__


      exception cause


   .. py:class:: __context__


      exception context


   .. py:class:: __suppress_context__



   .. py:class:: __traceback__



   .. py:class:: args



   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __setattr__()

      Implement setattr(self, name, value).


   .. py:method:: __setstate__()


   .. py:method:: __sizeof__()

      Size of object in memory, in bytes.


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



.. py:exception:: AuthError(message: str)


   Bases: :py:obj:`MapillaryException`

   Raised when a function is called without having the access token set in
   set_access_token to access Mapillary's API, primarily used in mapillary.set_access_token

   :var message: The error message returned
   :type message: str

   .. py:class:: __cause__


      exception cause


   .. py:class:: __context__


      exception context


   .. py:class:: __suppress_context__



   .. py:class:: __traceback__



   .. py:class:: args



   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __setattr__()

      Implement setattr(self, name, value).


   .. py:method:: __setstate__()


   .. py:method:: __sizeof__()

      Size of object in memory, in bytes.


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



.. py:exception:: InvalidImageResolutionError(resolution: int)


   Bases: :py:obj:`MapillaryException`

   Raised when trying to retrieve an image thumbnail with an invalid resolution/size.

   Primarily used with mapillary.image_thumbnail

   :var resolution: Image size entered by the user
   :type resolution: int

   .. py:class:: __cause__


      exception cause


   .. py:class:: __context__


      exception context


   .. py:class:: __suppress_context__



   .. py:class:: __traceback__



   .. py:class:: args



   .. py:method:: __str__() -> str

      Return str(self).


   .. py:method:: __repr__() -> str

      Return repr(self).


   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __setattr__()

      Implement setattr(self, name, value).


   .. py:method:: __setstate__()


   .. py:method:: __sizeof__()

      Size of object in memory, in bytes.


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



.. py:exception:: InvalidImageKeyError(image_id: Union[int, str])


   Bases: :py:obj:`MapillaryException`

   Raised when trying to retrieve an image thumbnail with an invalid image ID/key.
   Primarily used with mapillary.image_thumbnail

   :var image_id: Image ID/key entered by the user
   :param image_id: int

   .. py:class:: __cause__


      exception cause


   .. py:class:: __context__


      exception context


   .. py:class:: __suppress_context__



   .. py:class:: __traceback__



   .. py:class:: args



   .. py:method:: __str__() -> str

      Return str(self).


   .. py:method:: __repr__() -> str

      Return repr(self).


   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __setattr__()

      Implement setattr(self, name, value).


   .. py:method:: __setstate__()


   .. py:method:: __sizeof__()

      Size of object in memory, in bytes.


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



.. py:exception:: InvalidKwargError(func: str, key: str, value: str, options: list)


   Bases: :py:obj:`MapillaryException`

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

   .. py:class:: __cause__


      exception cause


   .. py:class:: __context__


      exception context


   .. py:class:: __suppress_context__



   .. py:class:: __traceback__



   .. py:class:: args



   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __setattr__()

      Implement setattr(self, name, value).


   .. py:method:: __setstate__()


   .. py:method:: __sizeof__()

      Size of object in memory, in bytes.


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



.. py:exception:: InvalidOptionError(param: str, value: any, options: list)


   Bases: :py:obj:`MapillaryException`

   Out of bound zoom error

   :var param: The invalid param passed
   :type param: str

   :var value: The invalid value passed
   :type value: any

   :var options: The possible list of zoom values
   :type options: list

   .. py:class:: __cause__


      exception cause


   .. py:class:: __context__


      exception context


   .. py:class:: __suppress_context__



   .. py:class:: __traceback__



   .. py:class:: args



   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __setattr__()

      Implement setattr(self, name, value).


   .. py:method:: __setstate__()


   .. py:method:: __sizeof__()

      Size of object in memory, in bytes.


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



.. py:exception:: InvalidFieldError(endpoint: str, field: list)


   Bases: :py:obj:`MapillaryException`

   Raised when an API endpoint is passed invalid field elements

   :var endpoint: The API endpoint that was targeted
   :type endpoint: str

   :var field: The invalid field that was passed
   :type field: list

   .. py:class:: __cause__


      exception cause


   .. py:class:: __context__


      exception context


   .. py:class:: __suppress_context__



   .. py:class:: __traceback__



   .. py:class:: args



   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __setattr__()

      Implement setattr(self, name, value).


   .. py:method:: __setstate__()


   .. py:method:: __sizeof__()

      Size of object in memory, in bytes.


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



.. py:exception:: LiteralEnforcementException(*args: object)


   Bases: :py:obj:`MapillaryException`

   Raised when literals passed do not correspond to options

   .. py:class:: __cause__


      exception cause


   .. py:class:: __context__


      exception context


   .. py:class:: __suppress_context__



   .. py:class:: __traceback__



   .. py:class:: args



   .. py:method:: enforce_literal(option_selected: str, options: Union[List[str], List[int]], param: str)
      :staticmethod:


   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __setattr__()

      Implement setattr(self, name, value).


   .. py:method:: __setstate__()


   .. py:method:: __sizeof__()

      Size of object in memory, in bytes.


   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



.. py:exception:: InvalidNumberOfArguments(number_of_params_passed: int, actual_allowed_params: int, param: str, *args: object)


   Bases: :py:obj:`MapillaryException`

   Raised when an inappropriate number of parameters are passed to a function

   .. py:class:: __cause__


      exception cause


   .. py:class:: __context__


      exception context


   .. py:class:: __suppress_context__



   .. py:class:: __traceback__



   .. py:class:: args



   .. py:method:: __str__()

      Return str(self).


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __delattr__()

      Implement delattr(self, name).


   .. py:method:: __dir__()

      Default dir() implementation.


   .. py:method:: __eq__()

      Return self==value.


   .. py:method:: __format__()

      Default object formatter.


   .. py:method:: __ge__()

      Return self>=value.


   .. py:method:: __getattribute__()

      Return getattr(self, name).


   .. py:method:: __gt__()

      Return self>value.


   .. py:method:: __hash__()

      Return hash(self).


   .. py:method:: __le__()

      Return self<=value.


   .. py:method:: __lt__()

      Return self<value.


   .. py:method:: __ne__()

      Return self!=value.


   .. py:method:: __reduce__()

      Helper for pickle.


   .. py:method:: __reduce_ex__()

      Helper for pickle.


   .. py:method:: __setattr__()

      Implement setattr(self, name, value).


   .. py:method:: __setstate__()


   .. py:method:: __sizeof__()

      Size of object in memory, in bytes.


   .. py:method:: __subclasshook__()

      Abstract classes can override this to customize issubclass().

      This is invoked early on by abc.ABCMeta.__subclasscheck__().
      It should return True, False or NotImplemented.  If it returns
      NotImplemented, the normal algorithm is used.  Otherwise, it
      overrides the normal algorithm (and the outcome is cached).


   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



