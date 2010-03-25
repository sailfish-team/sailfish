#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <GL/glut.h>

static PyObject* glVertexColorArray(PyObject *dummy, PyObject *args)
{
	PyObject *arg1 = NULL, *arg2 = NULL;
	PyObject *arr1 = NULL, *arr2 = NULL;

	if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2))
		return NULL;

	arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
	if (arr1 == NULL) return NULL;
	arr2 = PyArray_FROM_OTF(arg2, NPY_FLOAT, NPY_IN_ARRAY);
	if (arr2 == NULL) goto fail;

	int dimx = PyArray_DIM(arr1, 0);
	int dimy = PyArray_DIM(arr2, 1);
	int i;
	float *p0, *p1, *p2;

	glBegin(GL_QUADS);

	float *colors = PyArray_DATA(arr2);
	float *vertices = PyArray_DATA(arr1);

	for (i = 0; i < dimx; i++) {
		glColor3f(colors[3*i], colors[3*i+1], colors[3*i+2]);
		glVertex3f(vertices[3*i], vertices[3*i+1], vertices[3*i+2]);
/*		p0 = PyArray_GETPTR2(arr2, i, 0);
		p1 = PyArray_GETPTR2(arr2, i, 1);
		p2 = PyArray_GETPTR2(arr2, i, 2);
		glColor3f(*p0, *p1, *p2);

		p0 = PyArray_GETPTR2(arr1, i, 0);
		p1 = PyArray_GETPTR2(arr1, i, 1);
		p2 = PyArray_GETPTR2(arr1, i, 2);
		glVertex3f(*p0, *p1, *p2);
*/
	}

	glEnd();

	Py_DECREF(arr1);
	Py_DECREF(arr2);
	Py_INCREF(Py_None);
	return Py_None;

fail:
	Py_XDECREF(arr1);
	Py_XDECREF(arr2);
	return NULL;
}

static PyMethodDef mymethods[] = {
	{ "glVertexColorArray", glVertexColorArray,
	METH_VARARGS,
	"Render an OpenGL object using a vertex and color arrays"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initropengl(void)
{
	(void)Py_InitModule("ropengl", mymethods);
	import_array();
}



