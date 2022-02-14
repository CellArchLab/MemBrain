"""
Contains class Rigid3D for preforming rigid body transformation and scalar
scaling followed by translation on points (vectors).

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: rigid_3d.py 1461 2017-10-12 10:10:49Z vladan $

Note by Lorenz Lamm: This file is copied from Vladan Lucic in its entirety.
"""

__version__ = "$Revision: 1461 $"


from functools import partial

import numpy as np
import scipy as sp

from utils.affine import Affine

class Rigid3D(Affine):
    """

    Rotations in this class are such that points (vectors) are rotated and the
    coordinate system stays fixed. Obviously, a rotation that rotates
    the coordinate system and keeps points (vectors) fixed can be
    accomplished by changing the sign of the rotation angles.

    A rotation is specified by attributes:
      - q: roatation matrix
      - s_scalar: scale factor
      - gl: s*q (to be consistent with Affine)
      - s: scale matrix (to be consistent with Affine)
      - d: translation

    The order of axes of the array specifying coordinates is controlled by
    attribute / argument xy_axes. When xy_axes is 'dim_point', the
    coordinates array has dimensions n_dimensions x n_points. Conversly,
    ehn xy_axis is 'point_dim, the coordinates array has dimensions
    n_points x n_dimensions.

    Note that in this class xy_axes is always 'dim_point' except in
    methods that take xy_axes as an argument. This is  opposite from
    Affine where 'pint_dim' is the default, but both values can be used.

    """

    def __init__(self, q=None, scale=1., gl=None, d=None, order='qpsm'):
        """
        Sets self.order from arguments and self.xy_axes to 'dim_point' .

        """

        # initialize transformation parameters
        self.param_names = ['q', 'p', 's_scale', 'm', 'u', 'v']
        self.initializeParams()

        # parse arguments
        self.s_scalar = scale
        self.q = q
        if q is not None:
            self.q = np.asarray(q)
        elif gl is not None:
            self.gl = gl

        self.d = self.makeD(d, ndim=3)
        self.order = order

        # fixed, for now at least
        self.xy_axes = 'dim_point'


    ##############################################################
    #
    # Parameters
    #

    @classmethod
    def identity(cls, ndim=3):
        """
        Returnes an identity transformation.

        Argument ndim is ignored, it should be 2 here.
        """

        obj = cls(q=np.identity(3))
        return obj

    @classmethod
    def makeS(cls, scale):
        """
        Returns scale transformation in the matrix form corresponding to
        1D array scale.

        Argument:
          - scale: can be given as an 1d array (or a list), or as a single
          number in which case the scale is the same in all directions
        """

        s = cls.__base__.makeS(scale=scale, ndim=3)
        return s

    @classmethod
    def makeScalar(cls, s, check=True):
        """
        Returns scalar value corresponding to the scale matrix given by
        arg s.

        If arg check is True and arg s is not a proper isotropic scaling
        transformation (diagonal matrix having identical diagonal elements),
        ValueError is raised. If arg check is False, geometrical mean
        of all diagonal elements is returned.

        Arguments:
          - s: scale transformation matrix
          - check: flag indicating if matrix s is a proper isotropic scaling
          transformation (diagonal matrix having identical diagonal elements)
        """

        s00 = float(s[0,0])
        if check:
            if (s == s00 * np.identity(3)).all():
                s_scalar = s00
            else:
                raise ValueError(
                    "Martix s is not valid for a rigid transformation.")
        else:
            s_scalar = (s[0,0] * s[1,1] * s[2,2]) ** (1/3.)

        return s_scalar

    @classmethod
    def makeP(cls, parity, axis=-1):
        """
        Returns parity matrix corresponding to arg parity.

        If parity is -1, the element of the parity matrix corresponding to
        axis is set to -1 (all other are 1).

        Arguments:
          - parity: can be 1 or -1
          - axis: axis denoting parity element that can be -1
        """

        p = cls.__base__.makeP(parity=parity, axis=axis, ndim=3)
        return p

    @classmethod
    def makeD(cls, d, ndim=3):
        """
        Returns d (translation) array corresponding to arg parity.

        Arguments:
          - d: (single number) translation
        """

        d = cls.__base__.makeD(d, ndim=ndim)
        return d

    def getGl(self):
        """
        Get gl
        """
        if (self.s_scalar is not None) and (self.q is not None):
            return self.s_scalar * self.q
        else:
            return None

    def setGl(self, gl):
        """
        Set gl.
        """
        if gl is None:
            return
        q, p, s, m = self.decompose(gl=gl, order='qpsm')
        if (s.diagonal() == s[0,0]).all():
            self.q = q
            self.s_scalar = s[0,0]
        else:
            raise ValueError(
                "Scale obtained from transformation martix gl is not "
                + "isotropic. This might be due to a numerical error, "
                + " but better avoid setting attribute gl in this class.")

    gl = property(
        fget=getGl, fset=setGl, doc='gl from Affine is the same as q here')

    def getS(self):
        """
        Get s
        """
        if self.s_scalar is not None:
            return self.makeS(self.s_scalar)
        else:
            return None

    def setS(self, s):
        """
        Set s
        """
        if s is None:
            return
        s00 = float(s[0,0])
        if (s == s00 * np.identity(3)).all():
            self.s_scalar = s00
        else:
            raise ValueError(
                "Martix s is not valid for a rigid transformation.")

    s = property(fget=getS, fset=setS, doc="Scale in matrix form")

    def getNdim(self):
        "Returns number of dimensions (3)"
        return 3

    def setNdim(self, ndim):
        "Sets number of dimentsions. The only allowed value is 3."
        if ndim != 3:
            raise ValueError(
                "Argument ndim was " + str(ndim) +
                " , but the only allowed value is 3.")

    ndim = property(fget=getNdim, fset=setNdim, doc='N dimensions (3)')

    def inverse(self):
        """
        Finds inverse transformation of this instance or of the transformation
        specified by args gl and d. Returns a new instance of this class.

        Essentially the same as Affine.inverse(subgl='q'). The inverse is
        calculated as follows:

          q_inv = q^-1
          d_inv = -q_inv d

        The error is calculated as:

          error_inv = -q_inv error

        If self.error exists, calculates error for the inverse transformation.

        Returns: the inverted transformation
        """
        inv = super(Rigid3D, self).inverse(subgl='q')
        return inv

    @classmethod
    def compose(cls, t_1, t_2):
        """
        Finds composition of transformations t_1 and t_2. The resulting
        transformation is the same as if first t_2 was applied on initial
        coordinates, and then t_1 was applied.

        Essentially the same as Affine.compose(subgl='q'). The composition
        is calculated as follows:

          t_1 t_2 (x) = q_1 q_2 (x) + q_1 (d_2) + d_1

        The estimated rms error of the composition is:

          sqrt(rms_1 ** 2 + (mean_scale_1 rms_2) ** 2)

        where mean_scale_1 is the geometrical mean of all t_1 scales. It is
        saved as attribute rmsErrorEst, Attributes error and rmsError are not
        defined.

        Arguments:
          - t_1, t-2: transformation objects

        Returns:
          - new instance of this class that contains the composition
        """
        com = cls.__base__.compose(t_1=t_1, t_2=t_2, subgl='q')
        return com


    #########################################################
    #
    # Methods inherited from Affine that are not needed
    #

    @classmethod
    def find(cls, x, y, scale=None, xy_axes='dim_point'):
        """
        Should be implemented
        """
        raise NotImplementedError()

    @classmethod
    def findTwoStep(cls, x, y, scale=None, xy_axes='dim_point'):
        """
        Probably no need to be implemented
        """
        raise NotImplementedError()

    @classmethod
    def findTranslation(cls, x, y, scale=None, xy_axes='dim_point'):
        """
        Probably not needed
        """
        raise NotImplementedError()

    @classmethod
    def removeMasked(cls):
        """
        Not needed
        """
        raise NotImplementedError()

    #########################################################
    #
    # Methods to find rigid transformations when incomplete marker
    # points coordinates are given
    #

    @classmethod
    def find_32(
            cls, x, y, scale=None, use_jac=True, mode='constr_ck',
            ninit=10, randome=False, einit=None, einit_dist=0.1,
            randoms=False, sinit=1., maxiter=1000, return_all=False):
        """
        Finds optimal 3D transformation consisting of rotation, scale
        (optional) and translation that transform initial point coordinates
        (arg x) to final coordinates (arg y) when only x and y but not z
        components of the fnal coordinates are specified.

        For the optimization, 3D rotation, translation (x and y coordinates)
        and scale (scalar) are found that minimize the sum of squared
        differences between points y and x and y components of transformed
        points x. 3D rotation is parametrized by Caley-Klein parameters
        (closely related to quaternions). The optimization is performed by
        function sp.optimize.minimize() under the constraints ensuring that
        Caley-Klein parameters are normalized to 1 and that each of them is
        between -1 and 1 and that the scale is larger than 0.

        The optimization procedure is typically repeated several times (arg
        ninit, but also randome and randoms) and the best solution is
        retained. These runs differ by inital parameter values. The reason
        is that the optimal solution is not found for some combinations of
        initial and final coordinates and initial parameter values. A
        limited number of runs (around 10) typically suficces to obtain
        the optimal solution.

        If arg scale is None, the optimal 3d rotation and scale are determined.
        If arg scale is specified only the optimal 3d rotation is determined.
        In the latter case args randoms and sint are not considered.

        If args randome and randoms are both False, only one optimization run
        is performed (two if einit is 'gl2'), regrdless of the specified value
        of arg ninit. Otherwise, ninit optimizations are run.

        If args einit or sinit are set to 'gl2', 2D affine transformation
        (only the GL(2) part) is calculated based on the x and y coomponents
        of the initial coordinates (arg x) and on the final coordinates
        (arg y). Singular value decomposition of the resulting GL(2)
        transformation yields two rotations and two scales, which approximates
        the 3D rotation (exact if all initial point lie on the same z-plane).
        Namely, the rotations correspond to the Euler rotations around z axis
        (phi and psi), while the scales can be related to the Euler rotation
        around x-axis (theta). Theere is an inherent degeneracy in this
        approach, because the same affine scales correspond to both
        +theta and -theta.

        Initial 3D rotation parameter values and the number of runs (with
        different initial parameters) is determined as follows:

          1) randome is False:
            - einit is None: default initial rotation
            - einit is specified: specified initial rotation
            - einit == 'gl2': at least two runs with initial rotations
            determined by two solutions of 2D affine transformation

          2) randome is True:
            - einit is None: ninit runs with random initial rotations
            - einit is specified: ninit runs with random initial rotations
            in the einit_dist neighborhood of einit
            - einit == 'gl2': ninit runs with initial rotations in the
            einit_dist neighborhood of the two solutions of 2D affine
            transformation

        Initial scale value(s) is (are) deteremined as follows (only if
        scale is None):

          1) randoms is False:
            - sinit is None: default initial scale (1)
            - sinit is specified: specified initial scale
            - sinit == 'gl2': initial scale determined by the solutions of
            2D affine transformation

          2) randoms is True:
            - sinit is None: ninit runs with random initial scale centered
            around 1 (Maxwell-Boltzmann distributed, scale 1)
            - sinit is specified: ninit runs with random initial scale
            centered around sinit (Maxwell-Boltzmann distributed, scale set
            to sinit)
            - sinit == 'gl2': ninit runs with random initial scale
            centered around the 2d affine (GL2) solution (Maxwell-Boltzmann
            distributed, scale set to that of the affine transformation

        In cases where initial rotations and/or scale are specified and
        multiple runs a re performed, the first run uses the specified
        initial conditions, while the other are random.


        """

        # check mode
        if mode != 'constr_ck':
            raise ValueError(
                "Sorry, the only mode currently implemented is 'constr_ck'.")

        # convert to cm coords
        x_cm = x.mean(axis=-1).reshape((3,1))
        y_cm = y.mean(axis=-1).reshape((2,1))
        x_prime = x - x_cm
        y_prime = y - y_cm

        # find initial conditions for e
        gl2 = False
        if einit is None:
            einit_loc = einit
        elif isinstance(einit, (np.ndarray, list)):
            einit_loc = einit
        elif isinstance(einit, str) and (einit == 'gl2'):
            gl2 = True
            einit_loc, s_gl22 = cls.approx_gl2_to_ck3(
                x=x_prime, y=y_prime, xy_axes='dim_point', ret='both')
        else:
            raise ValueError(
                "Argument einit " + str(einit) + " was not understood")

        # find initial conditions for s
        if scale is None:

            # scale need to be found
            if sinit is None:
                sinit_loc = sinit
            elif isinstance(sinit, (float, int)):
                sinit_loc = sinit
            elif isinstance(sinit, str) and (sinit == 'gl2'):
                if gl2:
                    sinit_loc = s_gl22
                else:
                    dummy, sinit_loc = cls.approx_gl2_to_ck3(
                        x=x_prime, y=y_prime, xy_axes='dim_point', ret='both')
            else:
                raise ValueError(
                    "Argument sinit " + str(sinit) + " was not understood")

        elif isinstance(sinit, (float, int)):

            # scale fixed
            randoms = False
            sinit_loc = None

        # adjust n iterations
        if (randome is False) and (randoms is False):
            if gl2: ninit_loc = 2
            else: ninit_loc = 1
        else:
            ninit_loc = ninit

        # optimize rotation and scale
        if not gl2:

            # standard (not gl2)
            res = cls.find_32_constr_ck_multi(
                x=x_prime, y=y_prime, cm=False, use_jac=use_jac,
                randome=randome, einit=einit_loc, einit_dist=einit_dist,
                scale=scale, randoms=randoms, sinit=sinit_loc, maxiter=maxiter,
                ninit=ninit_loc, return_all=return_all)
            if return_all:
                rigid_cm, all_rigid_cm = res
            else:
                best = res

        else:

            # gl2 so do for both possibilities
            ninit_1 = max(int(ninit_loc / 2), 1)
            res_1 = cls.find_32_constr_ck_multi(
                x=x_prime, y=y_prime, cm=False, use_jac=use_jac, ninit=ninit_1,
                randome=randome, einit=einit_loc[0], einit_dist=einit_dist,
                randoms=randoms, sinit=sinit_loc, maxiter=maxiter)
            ninit_2 = max(ninit_loc - ninit_1, 1)
            res_2 = cls.find_32_constr_ck_multi(
                x=x_prime, y=y_prime, cm=False, use_jac=use_jac, ninit=ninit_2,
                randome=randome, einit=einit_loc[1], einit_dist=einit_dist,
                randoms=randoms, sinit=sinit_loc, maxiter=maxiter)

            if return_all:
                rigid_cm_1, all_rigid_cm_1 = res_1
                rigid_cm_2, all_rigid_cm_2 = res_2
                all_rigid_cm = all_rigid_cm_1 + all_rigid_cm_2
            else:
                rigid_cm_1 = res_1
                rigid_cm_2 = res_2

            if rigid_cm_1.optimizeResult.fun <= rigid_cm_2.optimizeResult.fun:
                best = rigid_cm_1
            else:
                best = rigid_cm_2

        # get translation
        translation_2 = (
            y_cm - best.s_scalar * np.dot(best.q[0:2,:], x_cm))
        translation = np.vstack((translation_2, [0]))

        # calculate full y in original (non-center of mass) frame
        y_3 = best.transform(x=x, d=translation.squeeze(1))

        # save translation related attributes
        best.d = translation.reshape(3)
        best.y = y_3

        # ToDo: see about modifying and returning all

        return best

    @classmethod
    def find_32_constr_ck_multi(
            cls, x, y, scale=None, cm=False, use_jac=True,
            ninit=10, randome=False, einit=None, einit_dist=0.1,
            randoms=False, sinit=1., maxiter=1000, return_all=False):
        """
        Finds the best rigid transformation in 3D that transform points x
        (initial) into points y (final) when only the first two coordinates
        are given for y (and all three for x), among solutions obtained
        for multiple initial parameters. In this way, possible false minima
        are avoided

        Uses sp.optimize.minimize() function to find the optimal solution.

        If arg scale is None, the optimal 3d rotation and scale are determined.
        If arg scale is specified only the optimal 3d rotation is determined.
        In the latter case args randoms and sint are not considered.

        Initial 3D rotation parameter values and the number of runs (with
        different initial parameters) is determined as follows:

          1) randome is False:
            - einit is None: default initial rotation
            - einit is specified: specified initial rotation

          2) randome is True:
            - einit is None: ninit runs with random initial rotations
            - einit is specified: ninit runs with random initial rotations
            in the einit_dist neighborhood of einit

        Initial scale value(s) is (are) teremined as follows (only if
        scale is None):

          1) randoms is False:
            - sinit is None: default initial scale (1)
            - sinit is specified: specified initial scale

          2) randoms is True:
            - sinit is None: ninit runs with random initial scale centered
            around 1 (Maxwell-Boltzmann distributed, scale 1)
            - sinit is specified: ninit runs with random initial scale
            centered around sinit (Maxwell-Boltzmann distributed, scale set
            to sinit)

        In cases where initial rotations and/or scale are specified and
        multiple runs a re performed, the first run uses the specified
        initial conditions, while the other are random.

        ToDo: If args randome and randoms are both False, only one
        optimization run is performed, regrdless of the specified value
        of arg ninit. Otherwise, ninit optimizations are run.

        Arguments:
          - x: initial points coordinates (3 x n_points martix)
          - y: final points coordinates (2 x n_points matrix)
          - scale: if None the optimization also solves for (scalar) scale
          factor, otherwise scale fixed at the specified factor (so 1 for
          plane rigid body movement)
          - init: (list or ndarray) initial parameters, the first four are
          Caley-Klein rotation paprameters and the fifth (in case arg
          scale is None) is the scale
          - cm: if True, initial and final coordinates are transformed to
          their respective center of mass systems
          - use_jac: flag indicating if Jacobian is passed to the optimization
          function (recommended)
          - maxiter: maximum number of iterations for the optimization
          procedure (same as arg maxiter of sp.optimize.minimize())
          - ninit: number of initial conditions
          - randome: Flag indicating whether random initial conditions
          for angles are generated
          - einit: initial rotation
          - einit_dist: specifies a neighborhood of a given rotation ih which
          random rotations are generated
          - randoms: Flag indicating whether random initial conditions
          for scale are generated
          - sinit: initial scale
          - return_all: flag indication whether all transformations are
          returned

        Returns the best transformation (instance of this group) with
        attributes:
          - ql: general linear matrix (in this case gl = s q)
          - q: rotation matrix
          - ck: Caley-Klein parameters corresponding to q
          - s, s_scalar: scale in matrix and scalar form
          - y: transformed inital coordinates that include z
          - optimizeResult: object returned by the optimization function
          (sp.optimize.minimize()) used
          - error: matrix of square errors (dimensions as arg y)
        """

        # default initial values
        default_e = np.array([1.,0,0,0])
        default_s = 1.

        # find best solution from random initial values
        best = None
        all = []
        for init_ind in range(ninit):

            # initial ck params
            if randome:
                if einit is None:

                    # totally random ck params
                    one_einit = cls.make_random_ck(center=None)

                else:

                    # random around initial ck params
                    if init_ind == 1:
                        one_einit = einit
                    else:
                        one_einit = cls.make_random_ck(
                            center=einit, distance=einit_dist)

            else:

                # single param set, default or specified
                if einit is None:
                    one_einit = default_e
                else:
                    one_einit = einit

            # add initial scale to the initial ck params if needed
            if scale is None:

                # optimize scale; start from random or fixed
                if randoms:
                    if sinit is None:

                        # random around 1
                        one_s_init = sp.stats.maxwell.rvs(
                            loc=0, scale=default_s)

                    else:

                        # random around init s
                        if init_ind == 1.:
                            one_s_init = sinit
                        else:
                            one_s_init = sp.stats.maxwell.rvs(
                                loc=0, scale=sinit)
                else:

                    # single param set, default or specified
                    if sinit is None:
                        one_s_init = default_s
                    else:
                        one_s_init = sinit

                # init params contain ck and scale
                one_init = np.hstack((one_einit, [one_s_init]))

            else:

                # don't optimize scale, init params only ck
                one_init = one_einit

            # solve and see if best solution so far
            rigid = cls.find_32_constr_ck(
                x=x, y=y, scale=scale, init=one_init, use_jac=use_jac,
                maxiter=maxiter)
            all.append(rigid)
            try:
                if rigid.optimizeResult.fun < best.optimizeResult.fun:
                    best = rigid
            except(NameError, AttributeError):
                best = rigid

        # return only the best solution or the best and all solutions
        if return_all:
            return best, all
        else:
            return best

    @classmethod
    def find_32_constr_ck(
            cls, x, y, scale=None, init=None, cm=False, use_jac=True,
            maxiter=1000):
        """
        Finds rigid transformation in 3D that transforms points x (initial)
        into points y (final) when only the first two coordinates are given
        for y and all three coordinates for x.

        Uses sp.optimize.minimize() function to find the optimal solution.

        If arg scale is None the optimization of transformation includes
        a search for scale. Otherwise, if arg scale is a number, scale is
        fixed at this value.

        If initial parameters are not specified (arg init is None), default
        initial parameters are used:
          - [1, 0, 0, 0, 1] if scale is None
          - [1, 0, 0, 0] if scale is bot None
        where the first four elements are the Caley-Klein parameters and the
        fifth (if existing) is the scale.

        After the transformation is calculated, it is applied to the initial
        coordinates (arg x) to yield transformed initial coordinates.

        Arguments:
          - x: initial points coordinates (3 x n_points martix)
          - y: final points coordinates (2 x n_points matrix)
          - scale: if None the optimization also solves for (scalar) scale
          factor, otherwise scale fixed at the specified factor (so 1 for
          plane rigid body movement)
          - init: (list or ndarray) initial parameters, the first four are
          Caley-Klein rotation paprameters and the fifth (in case arg
          scale is None) is the scale
          - cm: if True, initial and final coordinates are transformed to
          their respective center of mass systems
          - use_jac: flag indicating if Jacobian is passed to the optimization
          function (recommended)
          - maxiter: maximum number of iterations for the optimization
          procedure (same as arg maxiter of sp.optimize.minimize())

        Returns transformation (instance of this group) with attributes:
          - ql: general linear matrix (in this case gl = s q)
          - q: rotation matrix
          - ck: Caley-Klein parameters corresponding to q
          - s, s_scalar: scale in matrix and scalar form
          - y: transformed inital coordinates that include z
          - optimizeResult: object returned by the optimization function
          (sp.optimize.minimize()) used
          - error: matrix of square errors (dimensions as arg y)
        """

        # convert to CM coords
        if cm:
            x_prime = x - x.mean(axis=-1).reshape((3,1))
            y_prime = y - y.mean(axis=-1).reshape((2,1))
        else:
            x_prime = x
            y_prime = y

        # calculate matrices
        xxt = np.dot(x_prime, x_prime.transpose())
        yxt = np.dot(y_prime, x_prime.transpose())
        tryy = (y_prime * y_prime).sum()

        # function to mimimize
        sq_diff_ck = partial(
            cls.sq_diff_ck_23, scale=scale, xxt=xxt, yxt=yxt,
            make_r=cls.make_r_ck, const=tryy)

        # derivatives
        sq_diff_ck_deriv = partial(
            cls.sq_diff_ck_23_deriv, scale=scale, yxt=yxt, xxt=xxt,
            make_r=cls.make_r_ck, make_r_deriv=cls.make_r_ck_deriv)

        # check if should use derivatives
        if use_jac:
            jac = sq_diff_ck_deriv
        else:
            jac = None

        # constraints: e**2 = 1, e <= 1, e >= -1, s > 0
        if scale is None:

            # e normalized to 1, scale > 0
            constr_ck_norm = (
                {'type' : 'eq',
                 'fun' : lambda par: (par[:4]**2).sum() - 1,
                 'jac' : lambda par: np.hstack((np.asarray(2 * par[:4]), [0]))},
                {'type' : 'ineq',
                'fun' : lambda par: np.array(1 - par[0]),
                 'jac' : lambda par: np.array([-1,0,0,0,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 - par[1]),
                 'jac' : lambda par: np.array([0,-1,0,0,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 - par[2]),
                 'jac' : lambda par: np.array([0,0,-1,0,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 - par[3]),
                 'jac' : lambda par: np.array([0,0,0,-1,0])},
                {'type' : 'ineq',
                'fun' : lambda par: np.array(1 + par[0]),
                 'jac' : lambda par: np.array([1,0,0,0,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 + par[1]),
                 'jac' : lambda par: np.array([0,1,0,0,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 + par[2]),
                 'jac' : lambda par: np.array([0,0,1,0,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 + par[3]),
                 'jac' : lambda par: np.array([0,0,0,1,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(par[4]),
                 'jac' : lambda par: np.array([0,0,0,0,1])},
                )

        else:

            # e normalized to 1
            constr_ck_norm = (
                {'type' : 'eq',
                 'fun' : lambda par: (par**2).sum() - 1,
                 'jac' : lambda par: np.asarray(2 * par)},
                {'type' : 'ineq',
                'fun' : lambda par: np.array(1 - par[0]),
                 'jac' : lambda par: np.array([-1,0,0,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 - par[1]),
                 'jac' : lambda par: np.array([0,-1,0,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 - par[2]),
                 'jac' : lambda par: np.array([0,0,-1,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 - par[3]),
                 'jac' : lambda par: np.array([0,0,0,-1])},
                {'type' : 'ineq',
                'fun' : lambda par: np.array(1 + par[0]),
                 'jac' : lambda par: np.array([1,0,0,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 + par[1]),
                 'jac' : lambda par: np.array([0,1,0,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 + par[2]),
                 'jac' : lambda par: np.array([0,0,1,0])},
                {'type' : 'ineq',
                 'fun' : lambda par: np.array(1 + par[3]),
                 'jac' : lambda par: np.array([0,0,0,1])}
                )

        # init: [1, 0, 0, ...]
        if init is None:
            if scale is None:
                init = np.array([1.,0,0,0,1])
            else:
                init = np.array([1.,0,0,0])

        # solve
        res = sp.optimize.minimize(
            sq_diff_ck, init, jac=jac, constraints=constr_ck_norm,
            options={'disp': True, 'maxiter' : maxiter})

        # optimized parameters
        e_params = res.x[:4]
        if scale is None:
            s = res.x[4]
        else:
            s = scale

        # calculate missing r and y components
        r_33 = cls.make_r_ck(e_params)
        y_3 = s * np.dot(r_33[2,:], x_prime)
        y_33 = np.vstack((y_prime, y_3))

        # make instance
        inst = cls()
        #inst.gl = s * r_33
        inst.q = r_33
        inst.y = y_33
        inst.s_scalar = s
        inst.s = s * np.identity(3)
        inst.ck = e_params
        inst.optimizeResult = res
        inst.initial_params = init
        inst.error = y_prime - np.dot(inst.gl[:2,:], x_prime)
        #inst.rmsError = np.sqrt(np.square(inst.error).sum() / x_prime.shape[1])

        return inst

    @classmethod
    def sq_diff_ck_23(cls, param, scale, xxt, yxt, make_r, const):
        """
        Calculate sum of square differences using Cayley-Klein parameters:

          E = sum(y - s r x)^2

        using:

          E = tr( y^T y + s^2 x x^T r^T r - 2 s x y^T r)
        """

        # parameters
        e0, e1, e2, e3 = param[:4]
        if scale is None:
            s = param[4]
        else:
            s = scale

        # make r
        r_33 = make_r([e0, e1, e2, e3])
        r = r_33[:2,:]

        # square differences: s^2 tr(x x_t r_t r) - 2 s tr(x y_t r)
        rtr = np.dot(r.transpose(), r)
        term_1 = s**2 * (xxt * rtr).sum()
        term_2 = -2 * s * (yxt * r).sum()
        res = const + term_1 + term_2

        return res

    @classmethod
    def sq_diff_ck_23_deriv(
            cls, param, scale, yxt, xxt, make_r, make_r_deriv):
        """
        Calculates derivative (Jacobian) of sum of square differences
        in respect to Cayley-Klein parameters.

          dE / de_u = 2 ((r x x^T)_pi - (y x^T)_pi) dr_pi / de_u
          dE / ds = 2 tr(s x x^T r^T r - x y^T r)
        """

        # parameters
        e_params = param[:4]
        if scale is None:
            s = param[4]
        else:
            s = scale

        # prepare matrices
        r = make_r(e=e_params)[:2,:]
        dr_de = make_r_deriv(e=e_params)[:,:2,:]

        # calculate derivatives by e
        temp = 2 * s * (s * np.dot(r, xxt) - yxt)
        e_derivs = (temp * dr_de).sum(axis=1).sum(axis=1)

        # derivative by scale, if needed
        if scale is None:
            rtr = np.dot(r.transpose(), r)
            s_deriv = 2 * (s * (xxt * rtr).sum() - (yxt * r).sum())
            derivs = np.hstack((e_derivs, [s_deriv]))
        else:
            derivs = e_derivs

        return derivs

    @classmethod
    def approx_gl2_to_ck3(cls, x, y, xy_axes='dim_point', ret='both'):
        """
        Finds an approximate rigid (+scale) 3D transformation from 2D markers
        based on Gl(2) conversion to 3D rotation (and scale).

        The solution found has a degeneracy in the sign of theta. If arg
        re is 'one', only one rotation is returned, if it's 'both', both
        are returned.

        Arguments:
          - x: initial marker coordinates, need to have 3 spatial dimensions
          - y: transformed marker coordinates, 2 spatial dimensions
          - xy_axes: order of axes for x and y
          - ret: Flag indicating if one (value 'one') or both (value 'both')
          solutions

        Returns (e_param, scale):
          - e_param: Cayley-Klein parameters (ret=='one') or a list
          of the two solutions (for ret=='both')
          - scale: (scalar) scale
         """

        # remove 3rd dim from x
        if xy_axes == 'dim_point':
            x2 = x[:2,:]
        elif xy_axes == 'point_dim':
            x2 = x[:,:2]

        # find 2d affine
        from affine_2d import Affine2D
        affine22 = Affine2D.find(x2, y, xy_axes=xy_axes)

        # find corresponding ck params
        res = cls.gl2_to_ck3(gl=affine22.gl, ret=ret)

        return res

    @classmethod
    def gl2_to_ck3(cls, gl, ret='both'):
        """
        Finds Cayley-Klein parameters for a 3D rotation and scale that
        corresponsds to Gl(2) transformation gl.

        The parametrs found give (Euler) theta between -pi/2 and pi/2.

        The solution found has a degeneracy in the sign of theta. If arg
        re is 'one', only one rotation is returned, if it's 'both', both
        are returned.

        Arguments:
          - gl: Gl(2) matrix
          - ret: Flag indicating if one (value 'one') or both (value 'both')
          solutions

        Returns (e_param, scale):
          - e_param: ndarray of Cayley-Klein parameters (ret=='one') or a list
          of the two solutions (for ret=='both')
          - scale: (scalar) scale
        """

        # SV decompose gl
        from utils.affine import Affine
        from affine_2d import Affine2D
        aff = Affine()
        u, p, s, v = aff.decomposeSV(gl=gl)

        # convert to euler angles
        #v_angle = np.arctan2(v[1,0], v[0,0])
        #fi = v_angle
        fi = Affine2D.getAngle(v)
        s_diag = np.diag(s)
        s_scalar = s_diag.max()
        s2 = s_diag.min() / s_diag.max()
        theta = np.arccos(s2)
        #u_angle = np.arctan2(u[1,0], u[0,0])
        #psi = u_angle
        psi = Affine2D.getAngle(u)

        # euler angles to ck params
        e_param = cls.euler_to_ck(angles=[fi, theta, psi], mode='x')

        # get the other value if needed
        if ret == 'one':
            pass
        elif ret == 'both':
            e_param_2 = e_param.copy()
            e_param_2[1] = -e_param[1]
            e_param_2[2] = -e_param[2]
            e_param = [e_param, e_param_2]
        else:
            raise ValueError(
                "Could not understand argument ret: " + str(ret) + " . Valid "
                + "options are 'one' and 'both'")

        return e_param, s_scalar


    #########################################################
    #
    # Rotation-related matrices and parameters
    #

    @classmethod
    def make_r_ck(cls, e):
        """
        Returns active 3D rotation matrix parametrized by Cayley-Klein
        paramerters.

        The form is identical to the 3D rotation by quaternions given in
        Rotation in 3D Wikipedia where:
          (q_r, q_i, q_j, q_k) = (e_0, e_1, e_2, e_3).

        It also agrees with the rotation matrix parametrized by Euler
        / Cayley-Klein parameters given in Goldstein Classical Mechanics
        pg 153 except that the signs of e_1, e_2 and e_3 are changed
        because here we need the active and Goldstein gives the
        rotation in the passive form.
        """

        r = np.array(
            [[e[0]**2 + e[1]**2 - e[2]**2 - e[3]**2,
              2 * (e[1]*e[2] - e[0]*e[3]), 2 * (e[1]*e[3] + e[0]*e[2])],
             [2 * (e[1]*e[2] + e[0]*e[3]), e[0]**2 - e[1]**2 + e[2]**2 -e[3]**2,
              2 * (e[2]*e[3] - e[0]*e[1])],
             [2 * (e[1]*e[3] - e[0]*e[2]), 2 * (e[2]*e[3] + e[0]*e[1]),
              e[0]**2 - e[1]**2 - e[2]**2 + e[3]**2]])

        return r

    @classmethod
    def make_r_ck_deriv(cls, e):
        """
        Returns derivatives of 3D rotation matrix in respect to Cayley-Klein
        parameters.

        The returned value is 4x3x3 ndarray, where the first index denotes
        component of Cayley-Klein parameters in respect to which the
        derivatives are taken.
        """

        dr_de = 2 * np.array(
            # dr / de_0
            [[[e[0], -e[3], e[2]], [e[3], e[0], -e[1]], [-e[2], e[1], e[0]]],
             # dr / de_1
             [[e[1], e[2], e[3]], [e[2], -e[1], -e[0]], [e[3], e[0], -e[1]]],
             # dr / de_2
             [[-e[2], e[1], e[0]], [e[1], e[2], e[3]], [-e[0], e[3], -e[2]]],
             # dr / dr_3
             [[-e[3], -e[0], e[1]], [e[0], -e[3], e[2]], [e[1], e[2], e[3]]]]
        )

        return dr_de

    @classmethod
    def make_r_euler(cls, angles, mode='zxz_ex_active'):
        """
        Returns 3D rotation matrix parametrized by Euler angles.

        In the default mode (mode='zxz_ex_active'), the transformation
        is active (rotates points in space), extrinsic (rotations around axes
        of a fixed coordinate system) and the order or rotations applied is
        phi around z axis, theta around x and finally psi around z. The order
        of angles in arg angles is: phi, theta, psi.

        In this mode, the rotation matrix is transpose of what's given in
        Goldstein, Classical mechanics, pg 147 because the latter is for
        a passive, intrinsic XZX rotations (coodrdinate system is rotated and
        points in space are fixed).

        The mode 'zxz_in_active' is the same as the 'zxz_ex_active' except
        that it is intrinsic (the axes used for Euler rotations are rotated,
        not to be confused with passive rotations). The order of angles in
        arg angles is: phi, theta, psi.

        This form is the same as the one given on Euler angles Wikipedia
        (active, extrinsic), Z1 X2 Z3 form where angle_1 is psi,
        angle 2 theta and angle 3 phi, which is then the same as active
        intrinsic with angles ordered as phi, theta, psi.

        The mode 'zyz_ex_active' is active, extrinsic and the order of
        rotations is phi around z axis, theta around y and finally psi
        around z.

        The mode 'zyz_in_active' is active, intrinsic and the order of
        rotations is phi around z axis, theta around y and finally psi
        around z. This is the form recommended in Heymann, Chagoyen and
        Belnap JSB 2005 and used in Relion and XMIPP.

        An intrinsic rotation matrix can be obtained from the extrinsic
        matrix where the angles have opposite order (psi, theta, phi).
        That is to say, an intrinsic rotation can be obtained by using
        mode 'zxz_ex_active' and specifying angles as (psi, theta, phi).

        The corresponding passive modes are called 'zxz_ex_passive',
        'zxz_in_passive', 'zyz_ex_passive' and 'zyz_in_passive' (see
        convert_euler() for more info).

        Arguments:
          - angles: [phi, theta, psi] in rad
          - mode: Euler angles convention, currently implemented
          'zxz_ex_active', 'zxz_in_active', 'zyz_ex_active', 'zyz_in_active',
          'zxz_ex_passive', 'zxz_in_passive', 'zyz_ex_passive'
          and 'zyz_in_passive'

        Returns (ndarray) rotation matrix
        """

        # parse angles
        if (mode == 'x') or (mode == 'zxz_ex_active'):
            phi, theta, psi = angles
        #elif (mode == 'zxz_in_active'):
        #    psi, theta, phi = angles
        #elif mode == 'zyz_ex_active':
        #    phi, theta, psi = angles
        #    phi = phi - np.pi/2
        #    psi = psi + np.pi/2
        #elif mode == 'zyz_in_active':
        #    phi_arg, theta, psi_arg = angles
        #    phi = psi_arg - np.pi/2
        #    psi = phi_arg + np.pi/2

        #else:
        #    raise ValueError(
        #        "Mode " + mode + " is not defined. Currently implemented are "
        #        + "'zxz_ex_active' (same as 'x') and 'zxz_in_active'.")

        else:
            phi, theta, psi = Rigid3D.convert_euler(
                angles=angles, init=mode, final='zxz_ex_active')

        # rotation matrix for active extrinsic zxz Euler rotations
        #
        # Same as A in Goldstein pg 147 where all angles have the opposite
        # signs (because A is passive, intrinsic zxz, phi applied first)
        #
        # Same as Z_1 X_2 Z_3 on Euler angles wiki which is extrinsic with
        # angles (psi, theta, phi) or intrinsic with angles (phi, theta, psi)
        # in both cases active
        #
        # Check test_affine.testTransformArrayRigid3D()
        r = np.array(
            [[(np.cos(psi) * np.cos(phi)
               - np.cos(theta) * np.sin(phi) * np.sin(psi)),
              (-np.sin(phi) * np.cos(psi)
               - np.cos(theta) * np.sin(psi) * np.cos(phi)),
              np.sin(theta) * np.sin(psi)],
             [(np.cos(phi) * np.sin(psi)
               + np.cos(theta) * np.cos(psi) * np.sin(phi)),
              (-np.sin(psi) * np.sin(phi)
               + np.cos(theta) * np.cos(phi) * np.cos(psi)),
              -np.sin(theta) * np.cos(psi)],
             [np.sin(phi) * np.sin(theta),
              np.cos(phi) * np.sin(theta),
              np.cos(theta)]])
        return r

    @classmethod
    def convert_euler(cls, angles, init, final):
        """
        Converts Euler angles from the initial to the final Euler angle
        convention (mode).

        The following Euler angle conventions can be used: 'zxz_ex_active'
        'zxz_in_active', 'zyz_ex_active', 'zyz_in_active', 'zxz_ex_passive',
        'zxz_in_passive', 'zyz_ex_passive' and 'zyz_in_passive', where 'ex'
        and 'in' denote extrinsic and intrinsic transformations. These are
        defined in the standard way, namely:

        Active transformations rotate points (vectors) in space, while
        passive rotate the coordinate system and not points.

        A ZXZ transformation is composed of a rotation around z-axis,
        rotation around x-axis and another around z-axis. In a ZYZ
        transformation the second transformation is around y-axis.

        In case of extrinsic transformation, the axes corresponding to
        the three rotations are fixed. On the contrary, the (second and
        the third) rotations are carried around the rotated axes.

        The three Euler angles are given as (phi, theta, psi). in the
        order the rotations are applied.

        Note that the resulting angles are not necessarily in the proper
        range (phi and psi 0 - 2 pi and theta 0 - pi). However they can
        be used to make correct rotation matrices (using make_r_euler()).

        Arguments:
          - angles: (phi, theta, psi), where rotation by phi is applied first
          and by psi last
          - init: Euler angles convention (mode) in which arg. angles are
          specified
          - final: Euler angles convention (mode) to which arg. angles
          should be converted

        Returns Euler angles (phi, theta, psi) in the final Euler angle
        convention (mode).
        """

        # parse angles
        phi, theta, psi = np.asarray(angles, dtype=float)

        # check modes to avoid infinite recursions
        modes = [
            'x',
            'zxz_ex_active', 'zxz_in_active', 'zyz_ex_active', 'zyz_in_active',
            'zxz_ex_passive', 'zxz_in_passive', 'zyz_ex_passive',
            'zyz_in_passive']
        if init not in modes:
            raise ValueError(
                "Argument init: " + init + " was not understood. Defined "
                "values are given in the following list: " + str(modes))
        if final not in modes:
            raise ValueError(
                "Argument final: " + final + " was not understood. Defined "
                "values are given in the following list: " + str(modes))

        # initial mode zxz_ex_active
        if (init == 'x') or (init == 'zxz_ex_active'):

            # final mode active
            if (final == 'x') or (final == 'zxz_ex_active'):
                result = (phi, theta, psi)
            elif (final == 'zxz_in_active'):
                result = (psi, theta, phi)
            elif final == 'zyz_ex_active':
                result = (phi + np.pi/2, theta, psi - np.pi/2)
            elif final == 'zyz_in_active':
                result = (psi + np.pi/2, theta, phi - np.pi/2)

            elif final.endswith('_passive'):

                # final mode passive, convert via corresponding active
                active = final.split('_passive')[0] + '_active'
                intermed = cls.convert_euler(
                    angles=angles, init=init, final=active)
                result = (-intermed[2], -intermed[1], -intermed[0])

        # final mode zxz_ex_active
        elif (final == 'x') or (final == 'zxz_ex_active'):

            # initial mode active
            if (init == 'x') or (init == 'zxz_ex_active'):
                result = (phi, theta, psi)
            elif (init == 'zxz_in_active'):
                result = (psi, theta, phi)
            elif init == 'zyz_ex_active':
                result = (phi - np.pi/2, theta, psi + np.pi/2)
            elif init == 'zyz_in_active':
                result = (psi - np.pi/2, theta, phi + np.pi/2)

            elif init.endswith('_passive'):

                # init mode passive, convert via corresponding active
                active = init.split('_passive')[0] + '_active'
                intermed = cls.convert_euler(
                    angles=angles, init=active, final=final)
                result = (-intermed[2], -intermed[1], -intermed[0])

        # all other cases convert via 'zxz_ex_active'
        else:
            intermediate = cls.convert_euler(
                angles=angles, init=init, final='zxz_ex_active')
            result = cls.convert_euler(
                angles=intermediate, init='zxz_ex_active', final=final)

        return result

    @classmethod
    def make_r_axis(cls, angle, axis):
        """
        Returns 3D rotation matrix for a single rotation around one
        of the principal axes.

        Arguments:
          - angle: angle in rad
          - axis: 'x', 'y', or 'z'

        Returns 3D rotation matrix
        """

        if axis == 'z':
            r = np.array(
                [[np.cos(angle), -np.sin(angle), 0],
                 [np.sin(angle), np.cos(angle), 0],
                 [0, 0, 1]])

        elif axis == 'y':
            r = np.array(
                [[np.cos(angle), 0, np.sin(angle)],
                 [0, 1, 0],
                 [-np.sin(angle), 0, np.cos(angle)]])

        elif axis == 'x':
            r = np.array(
                [[1, 0, 0],
                 [0, np.cos(angle), -np.sin(angle)],
                 [0, np.sin(angle), np.cos(angle)]])

        else:
            raise ValueError(
                "Argument axis: ", axis, " was not understood. It should be "
                "'x', 'y', or 'z'.")

        return r

    @classmethod
    def extract_euler(cls, r, mode='zxz_ex_active', ret='one'):
        """
        Calculates Euler angles from a rotation matrix.

        There are two solutions, namely [phi, theta, psi] and
        [phi + pi, -theta, psi + pi]. If theta != 0, and arg ret is 'both',
        both solutions are returned. The first solution has positive theta
        (between 0 and pi) and the second has negative theta. If arg ret is
        'one', only the positive theta solution is returned.

        In case theta = 0 (degenerate case) the two solutions are
        [phi + psi, 0, 0] and [0, 0, phi + psi]. If arg ret is 'one', only
        the first solution is returned. If it is 'both', both solutions are
        returned.

        Arguments:
          - r: rotation matrix
          - mode: Euler angles convention, currently implemented
          'zxz_ex_active' and 'zxz_in_active'
          - ret: specifies is one ('one') or both solutions ('both')
          are returned

        Returns:
          - ret=='one': [phi, theta, psi]
          - ret=='both': [[phi_1, theta_1, psi_1], [phi_2, theta_2, psi_2]]
        """

        # calculate first for active xzx extrinsic
        theta_1 = np.arccos(r[2,2])
        theta_2 = -theta_1
        if theta_1 != 0:

            # non degenerate
            psi_1 = np.arctan2(
                r[0,2] / np.sin(theta_1), -r[1,2] / np.sin(theta_1))
            psi_2 = np.arctan2(
                r[0,2] / np.sin(-theta_1), -r[1,2] / np.sin(-theta_1))

            phi_1 = np.arctan2(
                r[2,0] / np.sin(theta_1), r[2,1] / np.sin(theta_1))
            phi_2 = np.arctan2(
                r[2,0] / np.sin(-theta_1), r[2,1] / np.sin(-theta_1))

        else:

            # degenerate case
            psi_1 = 0.
            phi_1 = np.arctan2(-r[0,1], r[0,0])

            phi_2 = 0.
            psi_2 = np.arctan2(r[1,0], r[0,0])

        # return
        if ret == 'one':

            if (mode == 'x') or (mode == 'zxz_ex_active'):
                return np.array([phi_1, theta_1, psi_1])
            elif (mode == 'zxz_in_active'):
                return np.array([psi_1, theta_1, phi_1])
            elif (mode == 'zyz_ex_active'):
                phi_1_ret = cls.shift_angle_range(
                    angle=phi_1 + np.pi/2, low=-np.pi)
                psi_1_ret = cls.shift_angle_range(
                    angle=psi_1 - np.pi/2, low=-np.pi)
                return np.array([phi_1_ret, theta_1, psi_1_ret])
            elif (mode == 'zyz_in_active'):
                psi_1_ret = cls.shift_angle_range(
                    angle=phi_1 + np.pi/2, low=-np.pi)
                phi_1_ret = cls.shift_angle_range(
                    angle=psi_1 - np.pi/2, low=-np.pi)
                return np.array([phi_1_ret, theta_1, psi_1_ret])

            else:
                raise ValueError(
                    "Mode " + mode + " is not defined. Currently implemented "
                    + "are 'zxz_ex_active' (same as 'x') and 'zxz_ex_active'.")

        elif ret == 'both':

            if (mode == 'x') or (mode == 'zxz_ex_active'):
                return np.array([[phi_1, theta_1, psi_1],
                                 [phi_2, theta_2, psi_2]])
            elif (mode == 'zxz_in_active'):
                return np.array([[psi_1, theta_1, phi_1],
                                 [psi_2, theta_2, phi_2]])
            elif (mode == 'zyz_ex_active'):
                phi_1_ret = cls.shift_angle_range(
                    angle=phi_1 + np.pi/2, low=-np.pi)
                psi_1_ret = cls.shift_angle_range(
                    angle=psi_1 - np.pi/2, low=-np.pi)
                phi_2_ret = cls.shift_angle_range(
                    angle=phi_2 + np.pi/2, low=-np.pi)
                psi_2_ret = cls.shift_angle_range(
                    angle=psi_2 - np.pi/2, low=-np.pi)
                return np.array([[phi_1_ret, theta_1, psi_1_ret],
                                 [phi_2_ret, theta_2, psi_2_ret]])
            elif (mode == 'zyz_in_active'):
                psi_1_ret = cls.shift_angle_range(
                    angle=phi_1 + np.pi/2, low=-np.pi)
                phi_1_ret = cls.shift_angle_range(
                    angle=psi_1 - np.pi/2, low=-np.pi)
                psi_2_ret = cls.shift_angle_range(
                    angle=phi_2 + np.pi/2, low=-np.pi)
                phi_2_ret = cls.shift_angle_range(
                    angle=psi_2 - np.pi/2, low=-np.pi)
                return np.array([[phi_1_ret, theta_1, psi_1_ret],
                                 [phi_2_ret, theta_2, psi_2_ret]])

    @classmethod
    def shift_angle_range(cls, angle, low=-np.pi):
        """
        Converts angle (in rad) so that it is between low and low + 2*pi.

        Arguments:
          - angle: angle (rad)
          - low: lower limit of the desired range, default -pi/2

        Returns converted angle (rad)
        """
        return np.mod(angle - low, 2*np.pi) + low

    @classmethod
    def euler_to_ck(cls, angles, mode='zxz_ex_active'):
        """
        Converts Euler angles to quaternions (Cayley-Klein parameters).

        The Euler angles can be given in any of the conventions (modes)
        defined in convert_euler().

        Arguments:
          - angles: Euler angles (phi, theta, psi) in rad
          - mode: Euler angles convention (mode) used for angles

        Returns: (e_0, e_1, e-2, e_3) Cayley-Klein parameters
        (rotation quaternions)
        """

        # unpack angles
        fi, theta, psi = np.asarray(angles, dtype=float)

        if (mode == 'x') or (mode == 'zxz_ex_active'):

            # This form is for active extrinsic zxz Euler angle convention.
            #
            # It is different from the one given in Goldstein pg 155
            # in that all Euler angles have the opposite sign, because
            # Goldstein uses passive intrinsic zxz.
            #
            # It is also different from

            res = np.array(
                [np.cos((fi + psi)/2.) * np.cos(theta/2.),
                 np.cos((-fi + psi)/2.) * np.sin(theta/2.),
                 np.sin((-fi + psi)/2.) * np.sin(theta/2.),
                 np.sin((fi + psi)/2.) * np.cos(theta/2.)]
                )

        elif mode == 'test':

            res = np.array(
                [np.cos((fi + psi)/2.) * np.cos(theta/2.),
                 np.cos((fi - psi)/2.) * np.sin(theta/2.),
                 np.sin((fi - psi)/2.) * np.sin(theta/2.),
                 np.sin((fi + psi)/2.) * np.cos(theta/2.)]
                )

        else:
            conv_angles = cls.convert_euler(
                angles=angles, init=mode, final='zxz_ex_active')
            res = cls.euler_to_ck(angles=conv_angles, mode='zxz_ex_active')

        return res

    @classmethod
    def make_random_ck(cls, center=None, distance=0.1):
        """
        Generates and returns Caley-Klein parameters for a random 3D rotation.

        If arg center is None, a random rotation is generated. These
        rotations are uniformly distributed (over unit S4 sphere).
        This is done by first generating a random 4D point in [-1,1].
        If the point lies outside the 4D unit sphere it is discarded.
        Otherwise, it is radially projected on the unit sphere to obtain
        a random 3D rotation.

        Alternatively, a random rotation is generated in the
        neighborhood of size (arg) distance, around the rotaition
        specified by arg center. In this case the

        If arg center is not None, arg distance needs to be specified.

        Arg distance is in "Caley-Klein parameter units". It should not be
        larger than 0.57, while the value of 0.1 corresponds roughly to 15 deg.

        Arguments:
          - center: (ndarray) 3D rotation parameters in terms of Caley-Klein
          parameters
          - distance: (float) size (radius) of the neighborhood of center
          (in "Caley-Klein parameter units")

        Returns Caley-Klein parameters for the generated 3D rotation.
        """

        if center is None:

            # make random inside 4-sphere and normalize
            while(True):
                e_random = np.random.random(4) * 2 - 1
                norm = np.sqrt(np.square(e_random).sum())
                if norm <= 1:
                    e_random = e_random / norm
                    break

        else:

            # random around initial
            e_small_123 = (np.random.random(3) * 2. - 1) * distance
            e_small_0 = np.sqrt(1 - np.square(e_small_123).sum())
            e_small = np.hstack(([e_small_0], e_small_123))
            center = np.asarray(center)
            r_e_random = np.dot(
                Rigid3D.make_r_ck(center), Rigid3D.make_r_ck(e_small))
            e_random_euler = Rigid3D.extract_euler(
                r_e_random, mode='x', ret='one')
            e_random = Rigid3D.euler_to_ck(e_random_euler, mode='x')

        return e_random

    @classmethod
    def angle_between_quat(cls, q1, q2):
        """
        Calculates angle between rotations given by quaternions q1 and q2.

        In other words, finds the angle of a rotation that transforms
        rotation q1 to q2. The axis for this rotation is not calculated.

        Arguments:
          - q1, q2: quaternion corresponding to the initial and final
          rotation, respectivly

        Returns: angle in rad (between 0 and pi)
        """

        dp = np.dot(q1, q2)
        theta = 2 * np.arccos(np.abs(dp))
        return theta

    @classmethod
    def angle_between_eulers(cls, angles1, angles2, mode='zxz_ex_active'):
        """
        Calculates angle between rotations where each rotation is
        specified by the three Euler angles.

        In other words, finds the angle of a rotation that transforms
        rotation angles1 to angles2. The axis for this rotation is not
        calculated.

        Arguments:
          - q1, q2: quaternion corresponding to the initial and final
          rotation, respectivly

        Returns: angle in rad (between 0 and pi)
        """

        # convert to quaternions
        q1 = cls.euler_to_ck(angles=angles1, mode=mode)
        q2 = cls.euler_to_ck(angles=angles2, mode=mode)

        # calculate angle
        res = cls.angle_between_quat(q1=q1, q2=q2)
        return res


    #########################################################
    #
    # Other methods
    #

    def transform(self, x, q=None, s=None, d=None, origin=None, xy_axes=None):
        """
        Applies transformation defined by q, s and d to points x.

        If q, s or d are None the corresponding attributes are used.

        If the arg xy_axes is 'point_dim' / 'dim_point', points used in this
        instance should be specified as n_point x 3 / 3 x n_point
        matrices.

        Arguments:
          - x: coordinates of one or more points
          - q: matrix representation of rotation
          - s: (single float or int) scale
          - d: translation vector (both 0 and None mean 0 in each coordinate)
          - origin: coordinates of the rotation and scale origin,
          None or 0 for origin at the coordinate system origin
          - xy_axes: order of axes in matrices representing points, can be
          'point_dim' or 'dim_point' (default)

        Returns:
          - transformed points in the same form as x, or None if x is None
          or has no elements
         """

        # set transformation params
        if xy_axes is None:
            xy_axes = self.xy_axes
        if q is None:
            q = self.q
        if s is None:
            s = self.s_scalar
        if d is None:
            d = self.d
        #if (d is None) or (np.isscalar(d) and (d == 0)):
        #    d = np.zeros(3, dtype='int')
        #elif isinstance(d, (list, np.ndarray)):
        #    d = np.asarray(d)
        #else:
        #    raise ValueError("Argument d: ", d, " was not understood.")

        # use Affine.transform()
        aff = Affine()
        aff.ndim = 3
        y = aff.transform(x=x, gl=s*q, d=d, origin=origin, xy_axes=xy_axes)

        # transformation
        #if xy_axes == 'point_dim':

            # equivalent to matrix multiplication of gl and transposed x,
            #y = s * np.inner(x, q) + d

        #elif xy_axes == 'dim_point':

            # just matrix product
            #y = s * np.dot(q, x) + np.expand_dims(d, 1)

        return y

    def recalculate_translation(self, rotation_center):
        """
        Recalculates translation when the current transformation is
        modified so that the rotation center is changed (not at the origin).

        The rotation center can be specified as 1d, 1x3 or 3x1 array. The
        returned translation will have the same form as the center.

        Argument:
          - rotation_center: (list or ndarray) new rotation center coordinates

        Returns translation
        """

        # check shape and axes order of rotation_center
        center = np.asarray(rotation_center, dtype='float')
        shape = center.shape
        if len(shape) == 1:
            xy_axes = 'dim'
            center = center.reshape((shape[0], 1))

        elif shape[0] == 1:
            xy_axes = 'point_dim'
            center = center.reshape((shape[1], 1))

        elif shape[1] == 1:
            xy_axes = 'dim_point'

        else:
            raise ValueError("Rotation center has to be 1d, 1x3 or 3x1 array")

        # make new translation
        trans = (self.transform(x=center, xy_axes='dim_point')
                 - self.s_scalar * center)

        # convert to the input form
        if xy_axes == 'dim':
            trans = trans.reshape(3)
        elif xy_axes == 'point_dim':
            trans = trans.reshape((1,3))

        return trans


    #########################################################
    #
    # Other approaches to find the optimal transformation.
    #
    # Should not be used.
    #
    # Probably implemented correctly but methods mostly don't work. Not
    # tested properly
    #

    @classmethod
    def _find_32_lsq(cls, x, y, init=None):
        """
        Not correct
        """

        raise NotImplementedError("Sorry, this is still work in progress.")

        # convert to CM coords
        x_cm = x - x.mean(axis=-1).reshape((3,1))
        y_cm = y - y.mean(axis=-1).reshape((2,1))

        #
        def residuals(params, x=x_cm, y=y_cm, make_r=cls.make_r_euler):
            """
            """

            # unpack variables
            fi, theta, psi = params

            # rotation matrix
            r = make_r(params)
            r_23 = r[0:2, :]

            # calculate residuals
            flat_shape = y.shape[0] * y.shape[1]
            res = (y_cm - np.dot(r_23, x_cm)).reshape(flat_shape)

            return res

        # initial
        if init is None:
            init = [0, 0, 0]

        # fit
        fit = sp.optimize.leastsq(residuals, init)

        return fit

    @classmethod
    def _find_32_constr(cls, x, y, init=None, cm=True):
        """
        Singular martix error
        """
        raise NotImplementedError("Sorry, this is still work in progress.")

        # convert to CM coords
        if cm:
            x_prime = x - x.mean(axis=-1).reshape((2,1))
            y_prime = y - y.mean(axis=-1).reshape((2,1))
        else:
            x_prime = x
            y_prime = y

        #
        xxt = np.dot(x_prime, x_prime.transpose())
        xyt = np.dot(x_prime, y_prime.transpose())
        yxt = np.dot(y_prime, x_prime.transpose())

        # function to minimize
        def sq_diff(r, xxt=xxt, xyt=xyt):
            """
            """

            r2d = r.reshape((2,3))
            rtr = np.dot(r2d.transpose(), r2d)

            # Tr(x * x_t * r_t * r)
            # perhaps (xtx * rtr).sum(), instead?
            term_1 = (
                np.inner(xxt[0,:], rtr[:,0]) + np.inner(xxt[1,:], rtr[:,1]) +
                np.inner(xxt[2,:], rtr[:,2]))

            # Tr(x * y_t * r)
            term_2 = (
                np.inner(xyt[0,:], r2d[:,0]) + np.inner(xyt[1,:], r2d[:,1]) +
                np.inner(xyt[2,:], r2d[:,2]))

            res = term_1 - 2 * term_2
            return res

        # derivative of the function to minimize
        def sq_diff_deriv(r, xxt=xxt, yxt=yxt):
            """
            """

            r2d = r.reshape((2,3))

            # r * x * x_t - 2 * y * x_t
            res2d = np.dot(r2d, xxt) - 2 * yxt
            res = res2d.reshape(6)

            return res

        # orthogonality constraint
        ortho = (
            {'type' : 'eq',
             'fun' : lambda r: np.array([r[0]**2 + r[1]**2 + r[2]**2 -1]),
             'jac' : lambda r: np.array([2*r[0], 2*r[1], 2*r[2], 0, 0, 0])},
            {'type' : 'eq',
             'fun' : lambda r: np.array([r[3]**2 + r[4]**2 + r[5]**2 -1]),
             'jac' : lambda r: np.array([2*r[0], 2*r[1], 2*r[2], 0, 0, 0])},
            {'type' : 'eq',
             'fun' : lambda r: np.array([r[0]*r[3] + r[1]*r[4] + r[2]*r[5]]),
             'jac' : lambda r: np.array([r[3], r[4], r[5], r[0], r[1], r[2]])})

        # use identity rotaion if initial not specified
        if init is None:
            init = np.array([1,0,0,0,1,0])

        # solve
        res = sp.optimize.minimize(
            sq_diff, init, jac=sq_diff_deriv, constraints=ortho,
            options={'disp': True})

        return res

    @classmethod
    def _find_22_constr(cls, x, y, init=None, cm=True, mode='fi'):
        """
        Mode 'r' doesn't work (singular matrix in LSQ)
        """
        raise NotImplementedError("Sorry, this is still work in progress.")

        # convert to CM coords
        if cm:
            x_prime = x - x.mean(axis=-1).reshape((2,1))
            y_prime = y - y.mean(axis=-1).reshape((2,1))
        else:
            x_prime = x
            y_prime = y

        # coordinates
        xyt = np.dot(x_prime, y_prime.transpose())
        yxt = np.dot(y_prime, x_prime.transpose())

        # not necessary, just to have correct min value
        trx2 = (x_prime * x_prime).sum()
        try2 = (y_prime * y_prime).sum()
        sq_diff_const = trx2 + try2

        # function to minimize
        def sq_diff_cfi_sfi(param, yxt=yxt, const=sq_diff_const):
            """
            """

            cfi, sfi = param
            r2d = np.array([[cfi, sfi], [-sfi, cfi]])

            # -2 * Tr(x y_t r) = -2 * (y x_t) * r
            res = const - 2 * (yxt * r2d).sum()
            return res

        # derivative of function to minimize
        def sq_diff_cfi_sfi_deriv(param, xyt=xyt):
            """
            """

            # -2 * [Tr(xty), -xyt[0,1] + xyt[1,0]]
            res = np.array([-2 * (xyt[0,0] + xyt[1,1]),
                            -2 * (-xyt[0,1] + xyt[1,0])])
            return res

        # constraint cfi**2 + sfi**2 = 1
        cons_cfi_sfi = (
            {'type' : 'eq',
             'fun' : lambda par: np.array([par[0]**2 + par[1]**2 - 1]),
             'jac' : lambda par: np.array([2 * par[0], 2 * par[1]])})

        def sq_diff_r(param, xyt=xyt, const=sq_diff_const):
            """
            """

            r2d = param.reshape((2,2))

            # -2 * Tr(x y_t r) = -2 * (y x_t) * r
            res = const - 2 * (xyt * r2d).sum()
            return res

        def sq_diff_r_deriv(param, xyt=xyt):
            """
            """

            # -2 * x y_t
            res = -2 * xyt.reshape(4)
            return res

        cons_r = (
            {'type' : 'eq',
             'fun' : lambda par: np.array([par[0]**2 + par[1]**2 - 1]),
             'jac' : lambda par: np.array([2 * par[0], 2 * par[1], 0, 0])},
            {'type' : 'eq',
             'fun' : lambda par: np.array([par[2]**2 + par[3]**2 - 1]),
             'jac' : lambda par: np.array([0, 0, 2 * par[2], 2 * par[2]])},
            {'type' : 'eq',
             'fun' : lambda par: np.array([par[0]*par[2] + par[1]*par[3]]),
             'jac' : lambda par: np.array([par[2], par[3], par[0], par[1]])},
            {'type' : 'eq',
             'fun' : lambda par: np.array([par[0]*par[3] - par[1]*par[2] - 1]),
             'jac' : lambda par: np.array([par[3], par[2], par[1], par[0]])}
            )

        # solve
        if mode == 'fi':

            # init
            if init is None:
                init = np.array([1, 0.])

            res = sp.optimize.minimize(
                sq_diff_cfi_sfi, init, jac=sq_diff_cfi_sfi_deriv,
                constraints=cons_cfi_sfi, options={'disp': True})

        elif mode == 'r':

            # init
            if init is None:
                init = np.array([1, 0., 0, 1])

            res = sp.optimize.minimize(
                sq_diff_r, init, jac=sq_diff_r_deriv,
                constraints=cons_r, options={'disp': True})

        return res

    @classmethod
    def _find_32_constr_ck_33(
            cls, x, y, init=None, cm=True, use_jac=True, maxiter=1000):
        """
        Doesn't always work
        """
        raise NotImplementedError("Sorry, this is still work in progress.")

        # convert to CM coords
        if cm:
            x_prime = x - x.mean(axis=-1).reshape((3,1))
            y_prime = y - y.mean(axis=-1).reshape((2,1))
        else:
            x_prime = x
            y_prime = y

        # coordinates
        #xyt = np.dot(x_prime, y_prime.transpose())
        #yxt = np.dot(y_prime, x_prime.transpose())

        # not necessary, just to have correct min value
        trxx = (x_prime * x_prime).sum()
        #try2 = (y_prime * y_prime).sum()
        #sq_diff_const = trx2 + try2

        # function to minimize
        def sq_diff_ck_33(
                param, x=x_prime, y=y_prime, make_r=cls.make_r_ck, const=trxx):
            """
            """

            # parameters
            e0, e1, e2, e3 = param[:4]
            y3 = param[4:]

            # make full y matrix
            y_3n = np.vstack([y, y3])

            # make r
            #r = np.array(
            #    [[e0**2 + e1**2 - e2**2 - e3**2,
            #      2 * (e1*e2 + e0*e3), 2 * (e1*e3 - e0*e2)],
            #     [2 * (e1*e2 - e0*e3), e0**2 - e1**2 + e2**2 - e3**2,
            #      2 * (e2*e3 + e0*e1)],
            #     [2 * (e1*e3 + e0*e2), 2 * (e2*e3 - e0*e1),
            #      e0**2 - e1**2 - e2**2 + e3**2]])
            r = make_r([e0, e1, e2, e3])

            # Tr(x_t x + y_t y - 2 x y_t r)
            yxt = np.dot(y_3n, x_prime.transpose())
            res = const + (y_3n * y_3n).sum() - 2 * (yxt * r).sum()

            return res

        def sq_diff_ck_33_deriv(
                param, x=x_prime, y=y_prime, make_r=cls.make_r_ck,
                make_r_deriv=cls.make_r_ck_deriv):
            """
            """

            # parameters
            e0, e1, e2, e3 = param[:4]
            y3 = param[4:]

            # make full y matrix
            y_3n = np.vstack([y, y3])

            # y x_t
            yxt = np.dot(y_3n, x.transpose())

            # make r
            #r = np.array(
            #    [[e0**2 + e1**2 - e2**2 - e3**2,
            #      2 * (e1*e2 + e0*e3), 2 * (e1*e3 - e0*e2)],
            #     [2 * (e1*e2 - e0*e3), e0**2 - e1**2 + e2**2 - e3**2,
            #      2 * (e2*e3 + e0*e1)],
            #     [2 * (e1*e3 + e0*e2), 2 * (e2*e3 - e0*e1),
            #      e0**2 - e1**2 - e2**2 + e3**2]])
            r = make_r([e0, e1, e2, e3])

            # dr / de
            #dr_de0 = 2 * np.array(
            #    [[e0, e3, -e2], [-e3, e0, e1], [e2, -e1, e0]])
            #dr_de1 = 2 * np.array(
            #    [[e1, e2, e3], [e2, -e1, e0], [e3, -e0, -e1]])
            #dr_de2 = 2 * np.array(
            #    [[-e2, e1, -e0], [e1, e2, e3], [e0, e3, -e2]])
            #dr_de3 = np.array(
            #    [[-e3, e0, e1], [-e0, -e3, e2], [e1, e2, e3]])
            dr_de0, dr_de1, dr_de2, dr_de3 = make_r_deriv([e0, e1, e2, e3])

            # d / de
            d_de0 = -2 * (yxt * dr_de0).sum()
            d_de1 = -2 * (yxt * dr_de1).sum()
            d_de2 = -2 * (yxt * dr_de2).sum()
            d_de3 = -2 * (yxt * dr_de3).sum()

            # d / dy3
            rx3 = np.dot(r[2,:], x)
            d_dy3 = 2 * (y_3n[2,:] - rx3)

            # together
            res = np.hstack(([d_de0, d_de1, d_de2, d_de3], d_dy3))
            return res

        # constraint e**2 = 1
        def constr_jac(param, i):
            """
            """
            res = np.zeros_like(param)
            res[i] = -1
            return res
        constr_ck_norm = (
            {'type' : 'eq',
             'fun' : lambda par: np.array([(par[:4]**2).sum() - 1]),
             'jac' : lambda par: 2 * np.hstack(
                 [par[:4], np.zeros(len(par) - 4)])},
            {'type' : 'ineq',
            'fun' : lambda par: np.array(1 - par[0]),
            'jac' : partial(constr_jac, i=0)},
             {'type' : 'ineq',
              'fun' : lambda par: np.array(1 - par[1]),
              'jac' : partial(constr_jac, i=1)},
             {'type' : 'ineq',
              'fun' : lambda par: np.array(1 - par[2]),
              'jac' : partial(constr_jac, i=2)},
            {'type' : 'ineq',
             'fun' : lambda par: np.array(1 - par[3]),
             'jac' : partial(constr_jac, i=3)}
            )

        # init: [1, 0, 0, ...]
        if init is None:
            init = np.zeros(4 + y_prime.shape[1])
            init[0] = 1.

        # jacobian
        if use_jac:
            jac = sq_diff_ck_33_deriv
        else:
            jac = None

        # solve
        res = sp.optimize.minimize(
            sq_diff_ck_33, init, jac=jac, constraints=constr_ck_norm,
            options={'disp': True, 'maxiter' : maxiter})

        return res
