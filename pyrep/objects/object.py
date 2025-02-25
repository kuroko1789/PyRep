from pyrep.backend import vrep
from pyrep.errors import *
from pyrep.const import ObjectType
from pyrep.errors import WrongObjectTypeError
from typing import List, Tuple, Union
import numpy as np


class Object(object):
    """Base class for V-REP scene objects that are used for building a scene.

    Objects are visible in the scene hierarchy and in the scene view.
    """

    def __init__(self, name_or_handle: Union[str, int]):
        if isinstance(name_or_handle, int):
            self._handle = name_or_handle
        else:
            self._handle = vrep.simGetObjectHandle(name_or_handle)
            assert_type = self.get_type()
            actual = ObjectType(vrep.simGetObjectType(self._handle))
            if actual != assert_type:
                raise WrongObjectTypeError(
                    'You requested object of type %s, but the actual type was '
                    '%s' % (assert_type.name, actual.name))

    def __eq__(self, other: 'Object'):
        return self.get_handle() == other.get_handle()

    @staticmethod
    def exists(name: str) -> bool:
        """Checks if the given object is in the scene.

        :param id: name/id of object. If the name is appended by a "@alt"
            suffix, then the object handle based on the object's alternative
            name will be retrieved.
        :return: True of the object exists.
        """
        try:
            vrep.simGetObjectHandle(name)
        except:
            return False
        return True

    @staticmethod
    def get_object_type(name: str) -> ObjectType:
        """Gets the type of the object.

        :return: Type of the object.
        """
        return ObjectType(vrep.simGetObjectType(vrep.simGetObjectHandle(name)))

    def get_type(self) -> ObjectType:
        """Gets the type of the object.

        :return: Type of the object.
        """
        raise NotImplementedError('Must be overridden.')

    def get_handle(self) -> int:
        """Gets the internal handle of this object.

        :return: The internal handle.
        """
        return self._handle

    def still_exists(self) -> bool:
        """Gets whether this object is still in the scene or not.

        :return: Whether the object exists or not.
        """
        return vrep.simGetObjectName(self._handle) != ''

    def get_name(self) -> str:
        """Gets the objects name in the scene.

        :return: The objects name.
        """
        return vrep.simGetObjectName(self._handle)

    def set_name(self, name: str) -> None:
        """Sets the objects name in the scene.
        """
        vrep.simSetObjectName(self._handle, name)

    def get_position(self, relative_to=None) -> List[float]:
        """Gets the position of this object.

        :param relative_to: Indicates relative to which reference frame we want
            the position. Specify None to retrieve the absolute position, or an
            Object relative to whose reference frame we want the position.
        :return: A list containing the x, y, z position of the object.
        """
        relto = -1 if relative_to is None else relative_to.get_handle()
        return vrep.simGetObjectPosition(self._handle, relto)

    def set_position(self, position: List[float], relative_to=None,
                     reset_dynamics=True) -> None:
        """Sets the position of this object.

        :param position: A list containing the x, y, z position of the object.
        :param relative_to: Indicates relative to which reference frame the
            the position is specified. Specify None to set the absolute
            position, or an Object relative to whose reference frame the
            position is specified.
        :param reset_dynamics: If we want to reset the dynamics when moving
            an object instantaneously.
        """
        relto = -1 if relative_to is None else relative_to.get_handle()
        if reset_dynamics:
            for ob in self.get_objects_in_tree(exclude_base=False):
                ob.reset_dynamic_object()

        vrep.simSetObjectPosition(self._handle, relto, position)

    def get_orientation(self, relative_to=None) -> List[float]:
        """Gets the orientation of this object.

        :param relative_to: Indicates relative to which reference frame we want
            the orientation. Specify None to retrieve the absolute orientation,
            or an Object relative to whose reference frame we want the
            orientation.
        :return: A list containing the x, y, z orientation of the
            object (in radians).
        """
        relto = -1 if relative_to is None else relative_to.get_handle()
        return vrep.simGetObjectOrientation(self._handle, relto)

    def set_orientation(self, orientation: List[float], relative_to=None,
                        reset_dynamics=True) -> None:
        """Sets the orientation of this object.

        :param orientation: A list containing the x, y, z orientation of
            the object (in radians).
        :param relative_to: Indicates relative to which reference frame the
            the orientation is specified. Specify None to set the absolute
            orientation, or an Object relative to whose reference frame the
            orientation is specified.
        :param reset_dynamics: If we want to reset the dynamics when rotating
            an object instantaneously.
        """
        relto = -1 if relative_to is None else relative_to.get_handle()
        if reset_dynamics:
            for ob in self.get_objects_in_tree(exclude_base=False):
                ob.reset_dynamic_object()
        vrep.simSetObjectOrientation(self._handle, relto, orientation)

    def get_quaternion(self, relative_to=None) -> List[float]:
        """Retrieves the quaternion (x,y,z,w) of an object.

        :param relative_to: Indicates relative to which reference frame we want
            the orientation. Specify None to retrieve the absolute orientation,
            or an Object relative to whose reference frame we want the
            orientation.
        :return: A list containing the quaternion (x,y,z,w).
        """
        relto = -1 if relative_to is None else relative_to.get_handle()
        return vrep.simGetObjectQuaternion(self._handle, relto)

    def set_quaternion(self, quaternion: List[float], relative_to=None,
                       reset_dynamics=True) -> None:
        """Sets the orientation of this object.

        If the quaternion is not normalised, it will be normalised for you.

        :param quaternion: A list containing the quaternion (x,y,z,w).
        :param relative_to: Indicates relative to which reference frame the
            the orientation is specified. Specify None to set the absolute
            orientation, or an Object relative to whose reference frame the
            orientation is specified.
        :param reset_dynamics: If we want to reset the dynamics when rotating
            an object instantaneously.
        """
        assert len(quaternion) == 4
        norm = np.linalg.norm(quaternion)
        if norm != 1.0:
            quaternion = list(np.array(quaternion) / norm)
        relto = -1 if relative_to is None else relative_to.get_handle()
        if reset_dynamics:
            for ob in self.get_objects_in_tree(exclude_base=False):
                ob.reset_dynamic_object()
        vrep.simSetObjectQuaternion(self._handle, relto, quaternion)

    def get_pose(self, relative_to=None) -> List[float]:
        """Retrieves the position and quaternion of an object

        :param relative_to: Indicates relative to which reference frame we want
            the pose. Specify None to retrieve the absolute pose, or an Object
            relative to whose reference frame we want the pose.
        :return: A list containing the (X,Y,Z,Qx,Qy,Qz,Qw) pose of the object.
        """
        p = self.get_position(relative_to)
        o = self.get_quaternion(relative_to)
        return p + o

    def set_pose(self, pose: List[float], relative_to=None,
                 reset_dynamics=True) -> None:
        """Sets the position and quaternion of an object.

        :param pose: A list containing the (X,Y,Z,Qx,Qy,Qz,Qw) pose of
            the object.
        :param relative_to: Indicates relative to which reference frame the
            the pose is specified. Specify None to set the absolute pose, or an
            Object relative to whose reference frame the pose is specified.
        :param reset_dynamics: If we want to reset the dynamics when rotating
            an object instantaneously.
        """
        assert len(pose) == 7
        self.set_position(pose[:3], relative_to, reset_dynamics)
        self.set_quaternion(pose[3:], relative_to, reset_dynamics)

    def get_parent(self) -> Union['Object', None]:
        """Gets the parent of this object in the scene hierarchy.

        :return: The parent of this object, or None if it doesn't have a parent.
        """
        try:
            handle = vrep.simGetObjectParent(self._handle)
        except RuntimeError:
            # Most probably no parent.
            return None
        return Object(handle)

    def set_parent(self, parent_object: Union['Object', None],
                   keep_in_place=True) -> None:
        """Sets this objects parent object in the scene hierarchy.

        :param parent_object: The object that will become parent, or None if
            the object should become parentless.
        :param keep_in_place: Indicates whether the object's absolute position
            and orientation should stay same
        """
        parent = -1 if parent_object is None else parent_object.get_handle()
        vrep.simSetObjectParent(self._handle, parent, keep_in_place)

    def get_matrix(self, relative_to=None) -> List[float]:
        """Retrieves the transformation matrix of this object.

        :param relative_to: Indicates relative to which reference frame we want
            the matrix. Specify None to retrieve the absolute transformation
            matrix, or an Object relative to whose reference frame we want the
            transformation matrix.
        :return: A list of 12 float values (the last row of the 4x4 matrix (
            0,0,0,1) is not needed).
                The x-axis of the orientation component is (m[0], m[4], m[8])
                The y-axis of the orientation component is (m[1], m[5], m[9])
                The z-axis of the orientation component is (m[2], m[6], m[10])
                The translation component is (m[3], m[7], m[11])
        """
        relto = -1 if relative_to is None else relative_to.get_handle()
        return vrep.simGetObjectMatrix(self._handle, relto)

    def set_matrix(self, matrix: List[float], relative_to=None) -> None:
        """Sets the transformation matrix of this object.

        :param relative_to: Indicates relative to which reference frame the
            matrix is specified. Specify None to set the absolute transformation
            matrix, or an Object relative to whose reference frame the
            transformation matrix is specified.
        :param matrix: A list of 12 float values (the last row of the 4x4 matrix
            (0,0,0,1) is not needed).
                The x-axis of the orientation component is (m[0], m[4], m[8])
                The y-axis of the orientation component is (m[1], m[5], m[9])
                The z-axis of the orientation component is (m[2], m[6], m[10])
                The translation component is (m[3], m[7], m[11])
        """
        relto = -1 if relative_to is None else relative_to.get_handle()
        vrep.simSetObjectMatrix(self._handle, relto, matrix)

    def is_collidable(self) -> bool:
        """Whether the object is collidable or not.

        :return: If the object is collidable.
        """
        return self._get_property(vrep.sim_objectspecialproperty_collidable)

    def set_collidable(self, value: bool) -> None:
        """Set whether the object is collidable or not.

        :param value: The new value of the collidable state.
        """
        self._set_property(vrep.sim_objectspecialproperty_collidable, value)

    def is_measurable(self) -> bool:
        """Whether the object is measurable or not.

        :return: If the object is measurable.
        """
        return self._get_property(vrep.sim_objectspecialproperty_measurable)

    def set_measurable(self, value: bool):
        """Set whether the object is measurable or not.

        :param value: The new value of the measurable state.
        """
        self._set_property(vrep.sim_objectspecialproperty_measurable, value)

    def is_detectable(self) -> bool:
        """Whether the object is detectable or not.

        :return: If the object is detectable.
        """
        return self._get_property(vrep.sim_objectspecialproperty_detectable_all)

    def set_detectable(self, value: bool):
        """Set whether the object is detectable or not.

        :param value: The new value of the detectable state.
        """
        self._set_property(vrep.sim_objectspecialproperty_detectable_all, value)

    def is_renderable(self) -> bool:
        """Whether the object is renderable or not.

        :return: If the object is renderable.
        """
        return self._get_property(vrep.sim_objectspecialproperty_renderable)

    def set_renderable(self, value: bool):
        """Set whether the object is renderable or not.

        :param value: The new value of the renderable state.
        """
        self._set_property(vrep.sim_objectspecialproperty_renderable, value)

    def is_model(self) -> bool:
        """Whether the object is a model or not.

        :return: If the object is a model.
        """
        prop = vrep.simGetModelProperty(self._handle)
        return not (prop & vrep.sim_modelproperty_not_model)

    def set_model(self, value: bool):
        """Set whether the object is a model or not.

        :param value: True to set as a model.
        """
        current = vrep.simGetModelProperty(self._handle)
        current |= vrep.sim_modelproperty_not_model
        if value:
            current -= vrep.sim_modelproperty_not_model
        vrep.simSetModelProperty(self._handle, current)

    def remove(self) -> None:
        """Removes this object/model from the scene.

        :raises: ObjectAlreadyRemoved if the object is no longer on the scene.
        """
        try:
            if self.is_model():
                vrep.simRemoveModel(self._handle)
            else:
                vrep.simRemoveObject(self._handle)
        except RuntimeError as e:
            raise ObjectAlreadyRemovedError(
                'The object/model was already deleted.') from e

    def reset_dynamic_object(self) -> None:
        """Dynamically resets an object that is dynamically simulated.

        This means that the object representation in the dynamics engine is
        removed, and added again. This can be useful when the set-up of a
        dynamically simulated chain needs to be modified during simulation
        (e.g. joint or shape attachement position/orientation changed).
        It should be noted that calling this on a dynamically simulated object
        might slightly change its position/orientation relative to its parent
        (since the object will be disconnected from the dynamics world in its
        current position/orientation), so the user is in charge of rectifying
        for that.
        """
        vrep.simResetDynamicObject(self._handle)

    def get_bounding_box(self) -> List[float]:
        """Gets the bounding box (relative to the object reference frame).

        :return: A list containing the min x, max x, min y, max y, min z, max z
            positions.
        """
        params = [vrep.sim_objfloatparam_objbbox_min_x,
                  vrep.sim_objfloatparam_objbbox_max_x,
                  vrep.sim_objfloatparam_objbbox_min_y,
                  vrep.sim_objfloatparam_objbbox_max_y,
                  vrep.sim_objfloatparam_objbbox_min_z,
                  vrep.sim_objfloatparam_objbbox_max_z]
        return [vrep.simGetObjectFloatParameter(
            self._handle, p) for p in params]

    def get_extension_string(self) -> str:
        """A string that describes additional environment/object properties.

        :return: The extension string.
        """
        return vrep.simGetExtensionString(self._handle, -1, '')

    def get_configuration_tree(self) -> bytes:
        """Retrieves configuration information for a hierarchy tree.

        Configuration includes object relative positions/orientations,
        joint/path values. Calling :py:meth:`PyRep.set_configuration_tree` at a
        later time, will restore the object configuration
        (use this function to temporarily save object
        positions/orientations/joint/path values).

        :return: The configuration tree.
        """
        return vrep.simGetConfigurationTree(self._handle)

    def rotate(self, rotation: List[float]) -> None:
        """Rotates a transformation matrix.

        :param rotation: The x, y, z rotation to perform (in radians).
        """
        m = vrep.simGetObjectMatrix(self._handle, -1)
        x_axis = [m[0], m[4], m[8]]
        y_axis = [m[1], m[5], m[9]]
        z_axis = [m[2], m[6], m[10]]
        axis_pos = vrep.simGetObjectPosition(self._handle, -1)
        m = vrep.simRotateAroundAxis(m, z_axis, axis_pos, rotation[2])
        m = vrep.simRotateAroundAxis(m, y_axis, axis_pos, rotation[1])
        m = vrep.simRotateAroundAxis(m, x_axis, axis_pos, rotation[0])
        vrep.simSetObjectMatrix(self._handle, -1, m)

    def check_collision(self, obj: 'Object' = None) -> bool:
        """Checks whether two entities are colliding.

        :param obj: The other collidable object to check collision against,
            or None to check against all collidable objects. Note that objects
            must be marked as collidable!
        :return: If the object is colliding.
        """
        handle = vrep.sim_handle_all if obj is None else obj.get_handle()
        return vrep.simCheckCollision(self._handle, handle) == 1

    # === Model specific methods ===

    def is_model_collidable(self) -> bool:
        """Whether the model is collidable or not.

        :raises: ObjectIsNotModel if the object is not a model.
        :return: If the model is collidable.
        """
        return self._get_model_property(
            vrep.sim_modelproperty_not_collidable)

    def set_model_collidable(self, value: bool):
        """Set whether the model is collidable or not.

        :param value: The new value of the collidable state of the model.
        :raises: ObjectIsNotModel if the object is not a model.
        """
        self._set_model_property(
            vrep.sim_modelproperty_not_collidable, value)

    def is_model_measurable(self) -> bool:
        """Whether the model is measurable or not.

        :raises: ObjectIsNotModel if the object is not a model.
        :return: If the model is measurable.
        """
        return self._get_model_property(
            vrep.sim_modelproperty_not_measurable)

    def set_model_measurable(self, value: bool):
        """Set whether the model is measurable or not.

        :param value: The new value of the measurable state of the model.
        :raises: ObjectIsNotModel if the object is not a model.
        """
        self._set_model_property(
            vrep.sim_modelproperty_not_measurable, value)

    def is_model_detectable(self) -> bool:
        """Whether the model is detectable or not.

        :raises: ObjectIsNotModel if the object is not a model.
        :return: If the model is detectable.
        """
        return self._get_model_property(
            vrep.sim_modelproperty_not_detectable)

    def set_model_detectable(self, value: bool):
        """Set whether the model is detectable or not.

        :param value: The new value of the detectable state of the model.
        :raises: ObjectIsNotModel if the object is not a model.
        """
        self._set_model_property(
            vrep.sim_modelproperty_not_detectable, value)

    def is_model_renderable(self) -> bool:
        """Whether the model is renderable or not.

        :raises: ObjectIsNotModel if the object is not a model.
        :return: If the model is renderable.
        """
        return self._get_model_property(
            vrep.sim_modelproperty_not_renderable)

    def set_model_renderable(self, value: bool):
        """Set whether the model is renderable or not.

        :param value: The new value of the renderable state of the model.
        :raises: ObjectIsNotModel if the object is not a model.
        """
        self._set_model_property(
            vrep.sim_modelproperty_not_renderable, value)

    def is_model_dynamic(self) -> bool:
        """Whether the model is dynamic or not.

        :raises: ObjectIsNotModel if the object is not a model.
        :return: If the model is dynamic.
        """
        return self._get_model_property(
            vrep.sim_modelproperty_not_dynamic)

    def set_model_dynamic(self, value: bool):
        """Set whether the model is dynamic or not.

        :param value: The new value of the dynamic state of the model.
        :raises: ObjectIsNotModel if the object is not a model.
        """
        self._set_model_property(
            vrep.sim_modelproperty_not_dynamic, value)

    def is_model_respondable(self) -> bool:
        """Whether the model is respondable or not.

        :raises: ObjectIsNotModel if the object is not a model.
        :return: If the model is respondable.
        """
        return self._get_model_property(
            vrep.sim_modelproperty_not_respondable)

    def set_model_respondable(self, value: bool):
        """Set whether the model is respondable or not.

        :param value: The new value of the respondable state of the model.
        :raises: ObjectIsNotModel if the object is not a model.
        """
        self._set_model_property(
            vrep.sim_modelproperty_not_respondable, value)

    def save_model(self, path: str) -> None:
        """Saves a model.

        Object can be turned to models via :py:meth:`Object.set_model`.
        Any existing file with same name will be overwritten.

        :param path: model filename. The filename extension is required ("ttm").
        :raises: ObjectIsNotModel if the object is not a model.
        """
        self._check_model()
        vrep.simSaveModel(self._handle, path)

    def get_model_bounding_box(self) -> List[float]:
        """Gets the models bounding box (relative to models reference frame).

        :raises: ObjectIsNotModel if the object is not a model.
        :return: A list containing the min x, max x, min y, max y, min z, max z
            positions.
        """
        self._check_model()
        params = [vrep.sim_objfloatparam_modelbbox_min_x,
                  vrep.sim_objfloatparam_modelbbox_max_x,
                  vrep.sim_objfloatparam_modelbbox_min_y,
                  vrep.sim_objfloatparam_modelbbox_max_y,
                  vrep.sim_objfloatparam_modelbbox_min_z,
                  vrep.sim_objfloatparam_modelbbox_max_z]
        return [vrep.simGetObjectFloatParameter(
            self._handle, p) for p in params]

    def get_objects_in_tree(self, object_type=ObjectType.ALL, exclude_base=True,
                            first_generation_only=False) -> List['Object']:
        """Retrieves the objects in a given hierarchy tree.

        :param object_type: The object type to retrieve.
            One of :py:class:`.ObjectType`.
        :param exclude_base: Exclude the tree base from the returned list.
        :param first_generation_only: Include in the returned list only the
            object's first children. Otherwise, entire hierarchy is returned.
        :return: A list of objects in the hierarchy tree.
        """
        options = 0
        if exclude_base:
            options |= 1
        if first_generation_only:
            options |= 2
        handles = vrep.simGetObjectsInTree(
            self._handle, object_type.value, options)
        objects = []
        for h in handles:
            objects.append(Object(h))
        return objects

    def copy(self) -> 'Object':
        """Copy and pastes object in the scene.

        The object is copied together with all its associated calculation
        objects and associated scripts.

        :return: The new pasted object.
        """
        return self.__class__((vrep.simCopyPasteObjects([self._handle], 0)[0]))

    def check_distance(self, other: 'Object') -> float:
        """Checks the minimum distance between two objects.

        :param other: The other object to check distance against.
        :return: The distance between the objects.
        """
        return vrep.simCheckDistance(
            self.get_handle(), other.get_handle(), -1)[6]

    # === Private methods ===

    def _check_model(self) -> None:
        if not self.is_model():
            raise ObjectIsNotModelError(
                "Object '%s' is not a model. Use 'set_model(True)' to convert.")

    def _get_model_property(self, prop_type: int) -> bool:
        current = vrep.simGetModelProperty(self._handle)
        return (current & prop_type) == 0

    def _set_model_property(self, prop_type: int, value: bool) -> None:
        current = vrep.simGetModelProperty(self._handle)
        current |= prop_type  # Makes is not X
        if value:
            current -= prop_type
        vrep.simSetModelProperty(self._handle, current)

    def _get_property(self, prop_type: int) -> bool:
        current = vrep.simGetObjectSpecialProperty(self._handle)
        return current & prop_type

    def _set_property(self, prop_type: int, value: bool) -> None:
        current = vrep.simGetObjectSpecialProperty(self._handle)
        current |= prop_type
        if not value:
            current -= prop_type
        vrep.simSetObjectSpecialProperty(self._handle, current)
