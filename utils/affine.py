"""
Contains class Affine for preforming affine transformation (general linear
transformation followed by translation) on points (vectors).

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: affine.py 1430 2017-03-24 13:18:43Z vladan $

Note by Lorenz Lamm: This file is copied from Vladan Lucic in its entirety.
"""

__version__ = "$Revision: 1430 $"

import warnings

import numpy
import scipy
import scipy.linalg as linalg


class Affine(object):
    """
    Finds and preforms affine transformation (general linear transformation
    followed by translation) on points (vectors) in an arbitrary
    dimensional space.

    The transformation that transforms points x to points y has the following
    form:

      y = gl x + d

    where:

      gl = q s p m

    Main methods:

      - find(): finds a transformation between two sets of points
      - findTwoStep(): finds a transformation between two sets of points in
      two steps
      - findTranslation(): finds a translation between two sets of points
      - decompose(): decomposes (factorizes) gl
      - composeGl(): makes gl from its decomposition, opposite of decompose()
      - identity(): makes the identity transformation
      - transform(): transforms a (set of) point(s)
      - transformArray: transforms array (image)
      - inverse(): calculates inverse transformation
      - compose(): composition of two transformations

    Transformation attributes (see formulas above):

      - d: translation vector
      - gl: general linear transformation matrix
      - q, u, v: rotation matrices
      - s: scaling matrix (diagonal, >=0)
      - scale: vector of scaling parameters (diagonal elements of s)
      - p: parity matrix (diagonal, the element self.parity_axis
      can be +1 or -1, other diagonal elements +1)
      - parity: parity (+1 or -1)
      - m: shear matrix (upper-triangular)
      - order: decomposition type. It is advisable not to change it directly
      (because this would make transformation attributes inconsistent with
      attribute order) but to set it via instantiation or change using
      decompose().

    Attributes related to finding a transformation:
      - error: error of transformation for all points
      - rmsError: root mean square error of the transformation
      - rmsErrorEst: estimate of rmsError, used when error can not be
      calculated directly, for example for transformations formed as a
      composition of two transformations (whose error or rmsError are known),
      see composition() method.

    Other attributes:
      - parity_axis: position of element in self.p that can be -1
      - param_names: (list) names of all transformation parameters
      - xy_axes: order of axes in matrices specifying points, can be
      'point_dim' (default, so for n points in d dimensions the points
      matrixs hape is nxd) or 'dim_point' (points shape is dxn).

    """

    ##############################################################
    #
    # Constants
    #

    # Axis that is flipped in case of negative parity
    parity_axis = -1

    ##############################################################
    #
    # Initialization
    #

    def __init__(self, gl=None, d=None, order='qpsm', xy_axes='point_dim'):
        """
        Sets self.gl to arg gl.

        If arg d is None, or 0, and gl is not None, self.d is set to
        numpy.array([0, 0, ...]) with the correct length. Otherwise self.d is
        set to arg d

        If the arg xy_axes is 'point_dim' / 'dim_point', points used in this
        instance should be specified as n_point x n_dim / n_dim x n_point
        matrices.

        Arguments:
          - gl: (numpy.ndarray of shape (ndim, ndim)) general linear
          transormation matrix
          - d: (numpy.ndarray of shape ndim) translation
          - order: decomposition order
          - xy_axes: order of axes in matrices representing points, can be
          'point_dim' (default) or 'dim_point'
        """

        # parse arguments
        self.xy_axes = xy_axes
        if gl is not None:
            self.gl = numpy.asarray(gl)
        else:
            self.gl = None
        # self.d = d
        # if (d is None) or (isinstance(d, numpy.int) and (d == 0)):
        #    if (self.gl is not None):
        #        self.d = numpy.zeros(gl.shape[0], dtype='int')
        self.d = self.makeD(d, ndim=self.ndim)

        # initialize order
        self.order = order

        # transformation parameters
        self.param_names = ['q', 'p', 's', 'm', 'u', 'v']

        # estimated error
        self.rmsErrorEst = None

    def initializeParams(self):
        """
        Sets all transformation parameters (q, p, s, m, u, v) to None.

        Transformation parameters are listed in self.param_names.
        """

        for name in self.param_names:
            self.__setattr__(name, None)

    ##############################################################
    #
    # Transformation parameters and error
    #

    @classmethod
    def identity(cls, ndim):
        """
        Returnes an identity object of this class, that is a transformation
        that leaves all vectors invariant.

        Argument:
          - ndim: number of dimensions
        """

        gl = numpy.identity(ndim)
        d = numpy.zeros(shape=ndim)
        obj = cls(gl=gl, d=d)

        return obj

    @classmethod
    def makeS(cls, scale, ndim):
        """
        Returns scale transformation in the matrix form corresponding to
        1D array scale.

        Arguments:
          - scale: can be given as an 1d array (or a list), or as a single
          number in which case the scale is the same in all directions
          - ndim: number of dimensions
        """

        if not isinstance(scale, (numpy.ndarray, list)):
            # make diagonal
            scale = numpy.ones(ndim) * scale

        # make matrix
        s = numpy.diag(scale)

        return s

    @classmethod
    def makeP(cls, parity, ndim, axis=-1):
        """
        Returns parity matrix corresponding to arg parity.

        If parity is -1, the element of the parity matrix corresponding to
        axis is set to -1 (all other are 1).

        Arguments:
          - parity: can be 1 or -1
          - axis: axis denoting parity element that can be -1
          - ndim: number of dimensions
        """

        # get ndim
        # if ndim is None:
        #    try:
        #        ndim = self.gl.shape[0]
        #    except AttributeError:
        #        ndim = self.q.shape[0]

        # get p
        p = numpy.identity(ndim)
        if parity == 1:
            pass
        elif parity == -1:
            p[axis, axis] = -1
        else:
            raise ValueError("Parity can be either 1 or -1.")
        return p

    @classmethod
    def makeD(cls, d, ndim=None):
        """
        Returns d (translation) array.

        If argument d is None, it is taken to be 0.

        If d is a single number, it is expanded to have the same translation
        in all dirrections, provided that arg ndim is specified. Otherwise,
        the same single number is returned.

        Important: ndim has to be specified, otherwise d array can not be
        made. Therefore, if arg ndim is None and arg d is None or a single,
        arg d can not be expanded.

        Arguments:
          - d: (single number) translation
          - ndim: number of dimensions

        Returns: (ndarray) translation vector or in case arg ndim is None
        and arg d is a single number or None, a single number is returned
        """

        # make d scalar if was None
        if (d is None):
            d = 0

        # try to expand scalar d to ndarray
        if numpy.isscalar(d):
            if ndim is not None:
                d = numpy.zeros(ndim) + d
        elif isinstance(d, (list, numpy.ndarray)):
            d = numpy.asarray(d)

        return d

    def getNdim(self):
        """
        """

        # try self._ndim
        try:
            ndim = self._ndim
            return ndim
        except AttributeError:
            pass

        #
        if self.gl is not None:
            ndim = self.gl.shape[0]
        else:
            ndim = None

        return ndim

    def setNdim(self, ndim):
        """
        Sets ndim
        """
        self._ndim = ndim

    ndim = property(fget=getNdim, fset=setNdim, doc='N dimensions')

    def getScale(self):
        """
        Extracts and returns scale. First tries to get scale from self.s. If
        self.s doesn't exist decomposes this transformation (self.decompose()).
        """
        try:
            rr = self.s
        except AttributeError:
            self.decompose()
        res = numpy.abs(self.s.diagonal())
        return res

    # scale = property(fget=getScale, doc='Scale vector')

    def setScale(self, scale):
        """
        Sets scale and (re)composes Gl with all other parameters unchanged

        Argument:
          - scale: (1d-array) scale
        """
        self.s = numpy.diag(scale)
        self.composeGl()

    scale = property(fget=getScale, fset=setScale, doc='scale')

    def getParity(self):
        """
        Extracts and returns parity. First tries to get parity from self.p. If
        self.p doesn't exist, calculates parity from det(self.gl).
        """
        try:
            pp = self.p
        except AttributeError:
            res = numpy.sign(linalg.det(self.gl))
        res = self.p.diagonal().prod()
        return res

    parity = property(fget=getParity, doc='Parity')

    def getTranslation(self):
        """
        Translation vector.
        """
        return self.d

    translation = property(fget=getTranslation, doc='Translation.')

    def getRMSError(self):
        """
        Root mean square of the error.

        First tries to calculate it from self.error. If self.error is not
        defined, returns self._rmsError or None if it doesn't exist)
        """
        try:
            if self.xy_axes == 'point_dim':
                n_points = self.error.shape[0]
            elif self.xy_axes == 'dim_point':
                n_points = self.error.shape[1]
            error = numpy.sqrt(numpy.square(self.error).sum() / float(n_points))
            return error
        except AttributeError:
            # try:
            #    return self._rmsError
            # except AttributeError:
            return None

    rmsError = property(fget=getRMSError, doc='Root mean square error')

    ##############################################################
    #
    # Finding and applying transformations
    #

    @classmethod
    def find(cls, x, y, type_='gl', order='qpsm', xy_axes='point_dim',
             x_ref='cm', y_ref='cm'):
        """
        Finds affine transformation (general linear transformation folowed by a
        translation) that minimizes square error for transforming points x to
        points y. The transformation has the form

          y = gl x + d,     gl = q s p m                                 (1)

        where d is translation vector and q, s, p and m are rotation, scaling,
        parity and shear matrices, respectivly.

        In 2D, if arg type_ is 'rs' (as opposed to the default 'gl'), instead
        of optimizing all parameters of Gl transformation, only rotation and
        one scale are optimized. See Affine2D.find() for more info.

        In the default mode (x_ref='cm' and y_ref='cm') the parameters are
        calculated by minimizing square error to get gl from:

          y - y_cm = gl (x - x_cm)   and   d = y_cm - gl x_cm

        where x_cm and y_cm are the centers of mass for x and y respectivly.
        In this case the square error of eq 1 is minimized

        In case args x_ref and y_ref are coordinates, gl is determined by
        minimizing square error in:

          y - y_ref = gl (x - x_ref)   and d = y_ref - gl x_ref

        Note that in this case the parameters found do not minimize the error
        of eq 1.

        In both cases general linear transformation (matrix gl) is calculated
        using scipy.linalg.lstsq().

        Only the points that are not masked neither in x_mask nor in y_mask are
        used. (probably not needed at all)

        Arguments:
          - x, y: sets of points, both having shape (n_points, n_dim)
          - x_ref, y_ref: (ndarray) coordinates of reference points, or 'cm' to
          use center of mass
          - xy_axes: indicates the order of axes in x and y; can be 'point_dim'
        so that x and y shape is n_points x n_dim or 'dim_point' for
        x_dim x n_points
          - type: transformation type ('gl' or 'rs')
          - order: gl decomposition order (see decompose())

        Returns the transformation found as an instance of class cls, with
        following attributes:
          - gl: general linear transformation matrix
          - d: translation vector
          - q, p, s, m: rotation, parity, scale and shear matrices
          - error: difference between y and transformed x values
          - resids, rank, singular: values returned from scipy.linalg.lstsq
          - xy_axes: same as arg xy_axes
          - _xPrime: x - x_ref
          - _yPrime: y - y_ref
          - type_: type of the optimization, 'gl' to find Gl transformation
          that optimizes the square error, or 'rs' to find the best rotation
          and one scale (currently implemented for 2D transformations only).
          In any case the translation is also found.
        """

        # remove masked points
        # [x, y], mask = cls.removeMasked([x,y], [x_mask,y_mask])
        # if (x_mask != None) or (y_mask != None):
        #    logging.warn("Arguments x_mask and y_mask are ignored.")

        # bring x and y to n_points x n_dim shape
        if xy_axes == 'point_dim':
            pass
        elif xy_axes == 'dim_point':
            x = x.transpose()
            y = y.transpose()
        else:
            raise ValueError(
                "Argument xy_axes was not understood. Possible values are: "
                + "'point_dim' and 'dim_point'.")

        # bring x to reference frame
        if isinstance(x_ref, str) and (x_ref == 'cm'):
            x_ref = numpy.mean(x, axis=0)
        elif isinstance(x_ref, (list, tuple, numpy.ndarray)):
            pass
        else:
            raise ValueError(
                'Argument x_ref: ', x_ref, ' was not understood.',
                " Allowed values are None, 'cm', or an array.")
        x_prime = x - x_ref

        # bring y to reference frame
        if isinstance(y_ref, str) and (y_ref == 'cm'):
            y_ref = numpy.mean(y, axis=0)
        elif isinstance(y_ref, (list, tuple, numpy.ndarray)):
            pass
        else:
            raise ValueError(
                'Argument y_ref: ', y_ref, ' was not understood.',
                " Allowed values are None, 'cm', or an array.")
        y_prime = y - y_ref

        # type_ should not be 'rs'
        if type_ == 'rs':
            warnings.warn(
                "Type 'rs' is not implemented for dimensions different from 2."
                + " Continuing with type 'gs'.")

        # find gl transformation
        gl_t, resids, rank, singular = linalg.lstsq(x_prime, y_prime)
        gl = gl_t.transpose()

        # find translation
        d = y_ref - numpy.inner(x_ref, gl)

        # instantiate and get error
        inst = cls(gl=gl, d=d)
        inst.resids = resids
        inst.rank = rank
        inst.singular = singular
        inst.xy_axes = xy_axes
        inst.error = y - inst.transform(x, xy_axes='point_dim')
        if xy_axes == 'dim_point':
            inst.error = inst.error.transpose()

        # save x and y in reference frame
        if xy_axes == 'point_dim':
            inst._xPrime = x_prime
            inst._yPrime = y_prime
        elif xy_axes == 'dim_point':
            inst._xPrime = x_prime.transpose()
            inst._yPrime = y_prime.transpose()

        # find and save other transformation matrices
        inst.decompose(order=order)

        # return
        return inst

    @classmethod
    def findTwoStep(cls, x, y, x_gl, y_gl, type_='gl', order='qpsm'):
        """
        Find affine transformation (like find()) in two steps. Useful when
        only few points x and y exist that are related by the full
        transformation (Gl and translation), but there are other points x_gl
        and y_gl which are related by a transformation having the same Gl but
        a different translation.

        In the first step, coordinates x_gl and y_gl are used to find Gl (see
        find() for details). In the second step, points Gl(x) and y are used
        to find the translation part (see findTranslation() for details.

        The final transformation is obtained by the composition of the two
        transformations obtained above. Consequently, attributes error and
        rmsError are not defined, but rmsErrorEst is.

        Returns new transformation.
        """

        # find Gl part of the transformation
        transf_gl = cls.find(x=x_gl, y=y_gl, type_=type_, order=order)
        transf_gl.d = numpy.zeros(shape=transf_gl.gl.shape[0])

        # find translation
        x_transf = transf_gl.transform(x=x, d=0)
        transf_d = cls.findTranslation(x=x_transf, y=y)

        # compose
        transf = cls.compose(t_1=transf_d, t_2=transf_gl)

        # save individual errors
        transf.glError = transf_gl.error
        transf.glRmsError = numpy.sqrt(
            numpy.square(transf.glError).sum() / float(transf.glError.shape[0]))
        transf.dError = transf_d.error
        transf.dRmsError = numpy.sqrt(
            numpy.square(transf.dError).sum() / float(transf.dError.shape[0]))

        return transf

    @classmethod
    def findTranslation(cls, x, y, x_mask=None, y_mask=None):
        """
        Finds translation between initial point (coordinates) x and final point

        Arguments:
          - x, y: sets of points, both having shape (n_points, n_dim)
          - x_mask, y_masks: masked (not used) points, vectors of length
          n_points

        Returns an instance of this class that has the calculated translation.
        All other tranformation parameters have identity values.
        """

        # make identity transform
        ndim = x.shape[1]
        inst = cls.identity(ndim=ndim)

        # remove masked points
        data, mask = cls.removeMasked(arrays=[x, y], masks=[x_mask, y_mask])
        x_unmasked = data[0]
        y_unmasked = data[1]

        # find centers of mass
        x_cm = numpy.mean(x_unmasked, axis=0)
        y_cm = numpy.mean(y_unmasked, axis=0)

        # find translation
        d = y_cm - x_cm
        inst.d = d
        inst.error = y_unmasked - inst.transform(x_unmasked)

        return inst

    def transform(self, x, gl=None, d=None, origin=None, xy_axes=None):
        """
        Applies transformation defined by gl and d to points x. The gl
        transformation is performed around arg origin.

        If args gl or d are None self.gl and self.d are used.

        If the arg xy_axes is 'point_dim' / 'dim_point', points (arg x)
        need to be specified as n_point x n_dim / n_dim x n_point
        matrices.

        If the arg xy_axes is 'mgrid', the input points (arg x) have
        to be in the dense mesh grid form, as returned by numpy.mgrid()
        and numpy.meshgrid(sparse=False).

        In all cases the returned points have the same form as the input
        points (arg x).

        Arguments:
          - x: coordinates of one or more points
          - gl: matrix representation of general linear transformation
          - d: translation vector (0 means 0 in each coordinate)
          - origin: coordinates of the origin for the gl transformation,
          None or 0 for origin at the coordinate system origin, otherwise
          list, ndarray or tuple
          - xy_axes: order of axes in matrices representing points

        Returns:
          - transformed points in the same form as x, or None if x is None
          or has no elements
        """
        # set gl and d
        if xy_axes is None:
            xy_axes = self.xy_axes
        if gl is None:
            gl = self.gl
        if d is None:
            d = self.d
        d = Affine.makeD(d, ndim=gl.shape[0])

        # odjust for origin if needed
        if ((origin is not None)
                and isinstance(origin, (list, numpy.ndarray, tuple))):
            gl_origin = numpy.identity(self.ndim) - gl
            d_origin = self.transform(
                x=origin, gl=gl_origin, d=0, xy_axes='point_dim')
            d = d + d_origin

        if (x is not None) and (len(x) > 0):

            if xy_axes == 'point_dim':

                # equivalent to matrix multiplication of gl and transposed x,
                res = numpy.inner(x, gl) + d

            elif xy_axes == 'dim_point':

                # just matrix product
                res = numpy.dot(gl, x) + numpy.expand_dims(d, 1)

            elif xy_axes == 'mgrid':

                # rotate: gl axis 1, x axis 0
                res = numpy.tensordot(gl, x, ([1], [0]))

                # translate
                d_exp = d
                for ax in range(len(d)):
                    d_exp = numpy.expand_dims(d_exp, -1)
                res = res + d_exp

        else:
            res = None

        return res

    def transformArray(
            self, array, origin, return_grid=False, output=None,
            order=1, mode='constant', cval=0.0, prefilter=False):
        """
        Transformes the given array, typically an image (arg array)
        according to the transformation of this instance. Rotation
        center is given by arg origin.

        Uses transform() method to (inversly) transform the complete index
        (coordinate) grid corresponding to the given array to make the
        transformed grid.

        Then it calls scipy.ndimage.map_coordinates() to transform the
        image according to the transformed grid. This step includes
        (n-linear or higer spline) interpolation.

        Note that in the default case (mode='constant') transformed values
        at coordinates that are even tiny bit outside the original grid are
        set to cval.

        Arguments:
          - array: array (image) to be transformed
          - origin: (1d ndarray) coordinates of the rotation origin
          - return_grid: flag indicating if the grid that was used to make
          this transformation is also returned
          - output, order, mode, cval, prefilter: arguments of
          scipy.ndimage.map_coordinates() (see that function for detailed
          info):
          - order: spline order, default 1 (n-linear)
          - prefilter: flag indicating if prefiltering is performed (needed
          for splines), default False
          - mode: how to deal with points outside boundaries, default 'constant'
          - cval: outside value for mode 'constant'

        Returns:
          - transformed array (image)
          - (optional) grid used for the transformation
        """

        # inverse transform original grid
        inverse = self.inverse()
        ori_grid = numpy.mgrid[tuple([slice(0, sha) for sha in array.shape])]
        new_grid = inverse.transform(ori_grid, origin=origin, xy_axes='mgrid')

        # transform image
        new_image = scipy.ndimage.map_coordinates(
            array, new_grid, output=output, order=order, mode=mode,
            cval=cval, prefilter=prefilter)

        # return
        if return_grid:
            return new_image, new_grid
        else:
            return new_image

    def _transformArray_old(
            self, array, origin, return_grid=False, output=None,
            order=1, mode='constant', cval=0.0, prefilter=False):
        """
        Transformes the given array, typically an image (arg array)
        according to the transformation of this instance. Rotation
        center is given by arg origin.

        Uses transform() method to (inversly) transform the complete index
        (coordinate) grid corresponding to the given array to make the
        transformed grid.

        Then it calls scipy.ndimage.map_coordinates() to transform the
        image according to the transformed grid. This step includes
        (n-linear or higer spline) interpolation.

        Note that in the default case (mode='constant') transformed values
        at coordinates that are even tiny bit outside the original grid are
        set to cval.

        Arguments:
          - array: array (image) to be transformed
          - origin: (1d ndarray) coordinates of the rotation origin
          - return_grid: flag indicating if the grid that was used to make
          this transformation is also returned
          - output, order, mode, cval, prefilter: arguments of
          scipy.ndimage.map_coordinates() (see that function for detailed
          info):
          - order: spline order, default 1 (n-linear)
          - prefilter: flag indicating if prefiltering is performed (needed
          for splines), default False
          - mode: how to deal with points outside boundaries, default 'constant'
          - cval: outside value for mode 'constant'

        Returns:
          - transformed array (image)
          - (optional) grid used for the transformation

        Note: To be removed
        """

        # original grid
        ori_grid = numpy.mgrid[tuple([slice(0, sha) for sha in array.shape])]

        # center original grid
        origin_expanded = origin
        for ax in range(len(origin)):
            origin_expanded = numpy.expand_dims(origin_expanded, -1)
        centered_grid = ori_grid - origin_expanded

        # grid transformation is inverse of intended image transform
        grid_t = self.inverse()

        # add translation that moves origin back
        ori_trans = self.identity(ndim=self.ndim)
        ori_trans.d = origin
        grid_t = self.compose(ori_trans, grid_t)

        # inverse transform centered grid
        new_grid = grid_t.transform(centered_grid, xy_axes='mgrid')

        # transform image
        new_image = scipy.ndimage.map_coordinates(
            array, new_grid, output=output, order=order, mode=mode,
            cval=cval, prefilter=prefilter)

        # return
        if return_grid:
            return new_image, new_grid
        else:
            return new_image

    ##############################################################
    #
    # Decomposing and composing Gl
    #

    def decompose(self, gl=None, order=None):
        """
        Decomposes gl using QR or singular value decomposition as follows:

          gl = q p s m (order 'qr' or 'qpsm')
          gl = p s m q (order 'rq' or 'psmq')
          gl = u p s v (order 'usv')

        where:
          - q, u, v: rotation matrix (orthogonal, with det +1)
          - p: parity matrix (diagonal, the element self.parity_axis can be +1
          or -1, other diagonal elements +1)
          - s: scale martix, diagonal and >=0
          - m: shear matrix, upper triangular, all diagonal elements 1

        The order is determined by agr oder. In this case self order is set to
        (arg) order). Otherwise, if arg order is None, self.order is used.

        Arguments:
          - gl: (ndarray) general linear transformation, or self.gl if None
          - order: decomposition order 'qpsm' (same as 'qr'), 'psmq' (same as
          'rq'), or 'usv'

        If arg gl is None, self.gl us used and the matrices resulting from the
        decomposition are saved as the arguments of this instance:
          - self.q, self.p, self.s and self.m if order 'qpsm', 'qr', 'psmq'
          or 'rq'
          - self.u, self.p, self.s, self.v if order 'usv'

        Returns only if gl is not None:
          - (q, p, s, m) if order 'qpsm', 'qr', 'psmq' or 'rq'
          - (u, p, s, v) if order 'usv'
        """

        # figure out gl (self.gl or arg) and the type of return
        if gl is None:
            gl = self.gl
            self.initializeParams()
            new = False
        else:
            new = True

        # figure our order
        if order is None:
            order = self.order
        else:
            self.order = order

        # set all transformation parameters to None
        # self.initializeParams()

        # call appropriate decompose method
        if ((order == 'qpsm') or (order == 'psmq') or (order == 'qr')
                or (order == 'rq')):
            q, p, s, m = self.decomposeQR(gl=gl, order=order)
            if new:
                return q, p, s, m
            else:
                self.q = q
                self.p = p
                self.s = s
                self.m = m

        elif (order == 'usv'):
            u, p, s, v = self.decomposeSV(gl=gl, order=order)
            if new:
                return u, p, s, v
            else:
                self.u = u
                self.p = p
                self.s = s
                self.v = v

        else:
            raise ValueError("Argument order: " + str(order) +
                             " not understood.")

    def decomposeQR(self, gl=None, order='qr'):
        """
        Decomposes gl using QR decomposition into:

          gl = q p s m (order 'qr' or 'qpsm')
          gl = p s m q (order 'rq' or 'psmq')

        where:
          - q: rotation (orthogonal, with det +1) matrix
          - p: parity (diagonal, all elements +1, except that the element
          corresponding to self.parity_axismatrix can be -1)
          possibly -1
          - s: scale martix, diagonal and positive
          - m: shear matrix, upper triangular, all diagonal elements 1

        Arguments:
          - gl: (ndarray) general linear transformation
          - order: decomposition order 'qr' or 'rq'

        Returns: (q, p, s, m)
        """

        # set decomposition type
        self.order = order

        # parse arg
        if gl is None:
            gl = self.gl
        ndim = gl.shape[0]

        # QR decompose
        if (order == 'rq') or (order == 'psmq'):
            r, q = linalg.rq(gl)
        elif (order == 'qr') or (order == 'qpsm'):
            q, r = linalg.qr(gl)
        else:
            ValueError("Argumnet order: ", order, " not understood. It should ",
                       "be 'psmq' (same as 'rq') or 'qpsm' (same as 'qr').")

        # extract s, p and m
        r_diag = r.diagonal()
        s_diag = numpy.abs(r_diag)
        s = numpy.diag(s_diag)
        p_diag = numpy.sign(r_diag)
        p = numpy.diag(p_diag)
        s_inv_diag = 1. * p_diag / s_diag
        m = numpy.dot(numpy.diag(s_inv_diag), r)

        # make q = q p and p = 1
        if (order == 'rq') or (order == 'psmq'):
            m = numpy.dot(numpy.dot(p, m), p)
            q = numpy.dot(p, q)
        elif (order == 'qr') or (order == 'qpsm'):
            q = numpy.dot(q, p)
        p = numpy.abs(p)

        # make sure det(q) > 0 and adjust p accordingly
        if linalg.det(q) < 0:
            p = numpy.identity(ndim, dtype=int)
            p[self.parity_axis, self.parity_axis] = -1
            if (order == 'rq') or (order == 'psmq'):
                q = numpy.dot(p, q)
                m = numpy.dot(numpy.dot(p, m), p)
            elif (order == 'qr') or (order == 'qpsm'):
                q = numpy.dot(q, p)

        return q, p, s, m

    def decomposeSV(self, gl, order='usv', correction='u'):
        """
        Decompose gl using singular value decomposition, so that:

          gl = u p s v

        where:
          - u, v: rotational matrices (orthogonal, det +1)
          - p: parity (diagonal, all elements +1, except that the element
          corresponding to self.parity_axis can be -1)
          - s: scale matrix (diagonal, all elements > 0)

        Arguments:
          - gl: general linear matrix
          - order: not implemented
          - correction: determines wheter u or v is adjusted in case p has
          more than one negative value

        Returns: (u, p, s, v)
        """

        # set decomposition type
        self.order = order

        # decompose
        u, s_diag, v = linalg.svd(gl)
        s = numpy.diag(s_diag)

        # get (initial) p from s and make all elements of s positive
        p_work = numpy.sign(s)
        s = numpy.abs(s)

        # make matrix that inverts parity
        ndim = gl.shape[0]
        invert_p = numpy.identity(ndim, dtype=int)
        invert_p[self.parity_axis, self.parity_axis] = -1

        # transform u and v so that their determinants are +1, if needed
        if linalg.det(u) < 0:
            p_work = numpy.dot(p_work, invert_p)
            u = numpy.dot(u, invert_p)
        if linalg.det(v) < 0:
            p_work = numpy.dot(p_work, invert_p)
            v = numpy.dot(invert_p, v)

        # split p into real p (at most one -1 element) and the correction
        parity = linalg.det(p_work)
        if parity > 0:
            p_corr = p_work
            p = numpy.diag(numpy.ones(ndim, dtype=int))
        else:
            p_corr = numpy.dot(p_work, invert_p)
            p = invert_p

        # sanity check
        if linalg.det(p_corr) < 0:
            raise ValueError("Something is wrong with parity")

        # adjust rotation matrices using the parity correction
        if correction == 'u':
            u = numpy.dot(u, p_corr)
        elif correction == 'v':
            v = numpy.dot(p_corr, v)
        else:
            raise ValueError("Argument correction: " + str(correction) +
                             " not understood. Allowed values are 'u' and 'v'.")

        return u, p, s, v

    def composeGl(self, order=None, q=None, p=None, s=None, m=None,
                  u=None, v=None):
        """
        Makes general linear transformation matrix (inverse of
        self.decompose()).

        If a parameter (q, p, s, m, u, or v) is not specified as argument, the
        corresponding attributes of this instance is used.

        If arg order is not specified, self.order is used.

        The parameters that are defined (passed as arguments or existing as
        attributes of this instance) have to correspond to the order. For
        example, if order is 'qpsm', parameters q, p, s and m have to be
        defined, or if order is 'usv' u, s, v and p have to be defined.

        Arguments:
          - q: rotation (orthogonal, with det 1) matrix
          - p: parity matrix, that is identity matrix with last element
          possibly -1
          - s: scale martix, diagonal and positive
          - m: shear matrix, upper triangular, all diagonal elements 1
          - u, v: singular value decomposition matrices

        Returns gl
        """

        # get order
        if order is None:
            order = self.order
        else:
            self.order = order

        # read parameters that were not passed
        if (order == 'qpsm') or (order == 'psmq'):
            if q is None: q = self.q
            if m is None: m = self.m
        elif order == 'usv':
            if u is None: u = self.u
            if v is None: v = self.v
        if p is None: p = self.p
        if s is None: s = self.s

        # compose
        if (order == 'qpsm') or (order == 'psmq'):
            ret = self.composeQR(order=order, q=q, p=p, s=s, m=m)
            self.order = order
        elif order == 'usv':
            ret = self.composeSV(order=order, u=u, p=p, s=s, v=v)
            self.order = order

        return ret

    def composeQR(self, order=None, q=None, p=None, s=None, m=None):
        """
        Makes general linear transformation from elements of 'qpsm' of
        'psmq' decomposition.
        """

        # parse arguments and fiqure out the return type
        if all([param is None for param in [q, p, s, m]]):
            new = False
        else:
            new = True

        # read parameters that were not passed
        if q is None: q = self.q
        if p is None: p = self.p
        if s is None: s = self.s
        if m is None: m = self.m

        # get order
        if order is None:
            order = self.order
        else:
            self.order = order

        # compose
        ps = numpy.dot(p, s)
        psm = numpy.dot(ps, m)
        if order == 'qpsm':
            gl = numpy.dot(q, psm)
        elif order == 'psmq':
            gl = numpy.dot(psm, q)

        # set or return
        if new:
            return gl
        else:
            self.gl = gl

    def composeSV(self, order=None, u=None, p=None, s=None, v=None):
        """
        Makes general linear transformation from elements of 'usv'
        decomposition.
        """

        # parse arguments and fiqure out the return type
        if all([param is None for param in [u, p, s, v]]):
            new = False
        else:
            new = True

        # read parameters that were not passed
        if u is None: u = self.u
        if p is None: p = self.p
        if s is None: s = self.s
        if v is None: v = self.v

        # get order
        if order is None:
            order = self.order
        else:
            self.order = order

        # compose
        ps = numpy.dot(p, s)
        ups = numpy.dot(u, ps)
        gl = numpy.dot(ups, v)

        # set or return
        if new:
            return gl
        else:
            self.gl = gl

    ##############################################################
    #
    # Operations of transformations
    #

    def inverse(self, gl=None, d=None, subgl=None):
        """
        Finds inverse transformation of this instance or of the transformation
        specified by args gl and d. Returns a new instance of this class.
        The inverse is calculated as follows:

          g_inv = g^-1
          d_inv = -g_inv d

        The error is calculated as:

          error_inv = -g_inv error

        If gl is not specified, and if self.error exists, calculates
        error for the inverse transformation.

        If gl is None self.gl and self.d are used (argument d is ignored).

        Arg subgl == 'q', should be used for the inverse of Rigid3D.

         Arguments:
          - gl: general linear transformation matrix
          - d: translation vector
          - subgl: specifies if argument different from gl is used to
          instantiate the inverse class. None to use gl, 'q' to use q (for
          Rigid3D)

        Returns: the inverted transformation
        """
        # set gl and d
        no_args = False
        if gl is None:
            gl = self.gl
            no_args = True
        if d is None:
            d = self.d
        if (d is None) or (numpy.isscalar(d) and (d == 0)):
            d = numpy.zeros(gl.shape[0], dtype='int')

        # calculate inverse
        gl_inv = linalg.inv(gl)
        d_inv = -numpy.dot(gl_inv, d)
        # d_inv = -self.transform(x=d, gl=gl_inv, d=0)

        # make new instance
        if subgl is None:
            tr_inv = self.__class__(gl=gl_inv, d=d_inv)
        elif subgl == 'q':
            q, p, s, m = self.decomposeQR(gl=gl_inv, order='qr')
            s_scalar = self.makeScalar(s, check=False)
            tr_inv = self.__class__(q=q, scale=s_scalar, d=d_inv)
        else:
            raise ValueError(
                "Argument subgl = " + subgl + " was not understood. Acceptable"
                + " values are None and 'q'.")

        # set ndim
        tr_inv.ndim = gl.shape[0]

        # try to invert error
        if no_args:
            try:
                #                tr_inv.error = -numpy.inner(gl_inv, self.error)
                tr_inv.error = -tr_inv.transform(self.error, d=0)
            except AttributeError:
                pass

        return tr_inv

    @classmethod
    def compose(cls, t_1, t_2, subgl=None):
        """
        Finds composition of transformations t_1 and t_2. The resulting
        transformation is the same as if first t_2 was applied on initial
        coordinates, and then t_1 was applied.

        The composition is calculated as follows:

          t_1 t_2 (x) = Gl_1 Gl_2 (x) + Gl_1 (d_2) + d_1

        The estimated rms error of the composition is:

          sqrt(rms_1 ** 2 + (mean_scale_1 rms_2) ** 2)

        where mean_scale_1 is the geometrical mean of all t_1 scales. It is
        saved as attribute rmsErrorEst, Attributes error and rmsError are not
        defined.

        Arg subgl == 'q', should be used for the composition of two instances
        of Rigid3D.

        Arguments:
          - t_1, t-2: transformation objects
          - subgl: specifies if argument different from gl is used to
          instantiate the inverse class. None to use gl, 'q' to use q (for
          Rigid3D)

        Returns:
          - new instance of this class that contains the composition
        """

        # calculate composition and make new instance
        gl = numpy.dot(t_1.gl, t_2.gl)
        # xy_axes='point_dim' needed because self.xy_axes can be different
        d = t_1.transform(x=t_2.d, xy_axes='point_dim')
        if subgl is None:
            tr = cls(gl=gl, d=d)
        elif subgl == 'q':
            aff = Affine(gl)
            q, p, s, m = aff.decomposeQR(order='qr')
            from utils.rigid_3d import Rigid3D
            s_scalar = Rigid3D.makeScalar(s, check=False)
            tr = Rigid3D(q=q, scale=s_scalar, d=d)
        else:
            raise ValueError(
                "Argument subgl = " + subgl + " was not understood. Acceptable"
                + " values are None and 'q'.")

        # get errors
        found_error = True
        try:
            t_1.rmsErrorEst
        except AttributeError:
            t_1.rmsErrorEst = None
        if t_1.rmsError is not None:
            t_1_rmsError = t_1.rmsError
        elif t_1.rmsErrorEst is not None:
            t_1_rmsError = t_1.rmsErrorEst
        else:
            found_error = False

        try:
            t_2.rmsErrorEst
        except AttributeError:
            t_2.rmsErrorEst = None
        if t_2.rmsError is not None:
            t_2_rmsError = t_2.rmsError
        elif t_2.rmsErrorEst is not None:
            t_2_rmsError = t_2.rmsErrorEst
        else:
            found_error = False

        # estimate rms error
        if found_error:
            # find scale
            q, p, s, m = t_1.decompose(order='qpsm', gl=t_1.gl)
            scale = s.diagonal()

            # estimate rms error
            mean_s1 = numpy.multiply.reduce(scale) ** (1. / len(scale))
            ms_error = t_1_rmsError ** 2 + (mean_s1 * t_2_rmsError) ** 2
            tr.rmsErrorEst = numpy.sqrt(ms_error)

        return tr

        ##############################################################

    #
    # Other methods
    #

    @classmethod
    def removeMasked(cls, arrays, masks=None):
        """
        Makes a mask that combine all masks and removes points from arrays that
        are masked by the mask.

        Arrays is a list of, a tuple of, or a single array, each containing
        coordinates of n_points points.

        Masks are a list of, a tuple of, or a single vector, each containing
        a mask for a respective element of arrays. Mask entry 0 means not
        masked, and 1 masked. If masks or any mask is None, it is understood
        as no mask.

        Arguments:
          - list of, tuple of, or single array, shape (n_points, ndim)
          - list of, tuple of, or single vector, length n_points

        Returns (arrays, mask):
          - list of, or a single array (depending on the argument) or
          - mask: combined mask (x_mask | y_mask)
        """

        # determine number of arrays
        if isinstance(arrays, numpy.ndarray):
            n_arrays = 1
            arrays = [arrays]
        elif isinstance(arrays, tuple) or isinstance(arrays, list):
            n_arrays = len(arrays)
        else:
            raise TypeError("Argument arrays can be ndarray, list or a tuple, "
                            + "but not " + type(arrays) + ".")

            # set masks if needed
        n_points = arrays[0].shape[-2]
        no_mask = numpy.zeros(n_points, dtype='bool')
        if masks is None:
            masks = [no_mask] * n_arrays
        expanded_masks = []

        # replace None masks
        for mas in masks:
            if mas is None:
                expanded_masks.append(no_mask)
            else:
                expanded_masks.append(mas)

        # combine masks
        total_mask = reduce(numpy.logical_or, expanded_masks)

        # remove masked points
        masked_data = [arr.compress(numpy.equal(total_mask, 0), axis=-2)
                       for arr in arrays]

        # return
        if len(masked_data) == 1:
            masked_data = masked_data[0]
        return masked_data, total_mask

