{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "EeNvZUYY1NGh",
    "outputId": "77bc9fe7-23b9-4b41-b9f5-41b467b0a33b"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1 2 3\n",
      "<class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>\n",
      "(5,) (2, 5) (2, 2, 5)\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a=np.array([1,2,3,4,5])\n",
    "b=np.array([[1,2,3,4,5],[6,7,8,9,10]])\n",
    "c=np.array([[[1,2,3,4,5],[6,7,8,9,10]],[[1,2,3,4,5],[6,7,8,9,10]]])\n",
    "print(a.ndim,b.ndim,c.ndim)\n",
    "print(type(a),type(b),type(c))\n",
    "print(np.shape(a),np.shape(b),np.shape(c))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1 2 3]\n",
      "  [4 5 6]]\n",
      "\n",
      " [[7 8 9]\n",
      "  [4 5 6]]]\n",
      "3\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[4,5,6]]])\n",
    "print(a)\n",
    "np.shape(a)\n",
    "print(a.ndim)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "BVao7gjMIr8b",
    "outputId": "736bef9c-4a79-4a2d-ba25-e5199fbf6d61"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n",
      "[[ 1.  2.  3.  4.  5.]\n",
      " [ 6.  7.  8.  9. 10.]]\n",
      "1.0\n",
      "6.0\n",
      "2.0\n",
      "7.0\n",
      "3.0\n",
      "8.0\n",
      "4.0\n",
      "9.0\n",
      "5.0\n",
      "10.0\n",
      "---------\n",
      "[[ 1.  2.  3.  4.  5.]\n",
      " [ 6.  7.  8.  9. 10.]]\n",
      "1.0\n",
      "2.0\n",
      "3.0\n",
      "4.0\n",
      "5.0\n",
      "6.0\n",
      "7.0\n",
      "8.0\n",
      "9.0\n",
      "10.0\n"
     ]
    }
   ],
   "source": [
    "#arrays with existing data F=column major, C-Row major order\n",
    "a1=np.array([[1,2,3,4,5],[6,7,8,9,10]])\n",
    "print(a1.ndim)\n",
    "a=np.asarray(a1,float,'F')\n",
    "print(a)\n",
    "for i in np.nditer(a):\n",
    "  print(i)\n",
    "print('---------')\n",
    "a=np.asarray(a1,float,'C')\n",
    "print(a)\n",
    "for i in np.nditer(a):\n",
    "  print(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "zPjhf4KrKPp2",
    "outputId": "31eb10b0-e676-4021-b140-e260d3889bd3"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[b'we' b'lc' b'om' b'e ']\n",
      "[1. 2. 3. 4.]\n"
     ]
    }
   ],
   "source": [
    "#string will be considered as a sequence of byte\n",
    "s=b\"welcome \"\n",
    "s1=np.frombuffer(s,\"S2\")\n",
    "print(s1)\n",
    "a1=[1,2,3,4,5]\n",
    "a=np.fromiter(a1,'float',4)\n",
    "print(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "sll2n84iLwhd",
    "outputId": "c51e8f65-8071-4715-e1fc-c13be5c6a401"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0. 2. 4. 6. 8.]\n",
      "(array([ 0.,  5., 10., 15., 20.]), 5.0)\n"
     ]
    }
   ],
   "source": [
    "#arrays with numerical ranges\n",
    "n=np.arange(0,10,2,'float')\n",
    "print(n)\n",
    "#np.linspace(startindex,stopindex,no of values,end point,return step,dtype)\n",
    "n1=np.linspace(0,20,5,True,True,float)\n",
    "print(n1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "k9fvNlzHN34C",
    "outputId": "7a0f8299-7dbd-4f90-f7a3-090b3c55a6e6"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1 2 3\n",
      "<class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>\n",
      "(5,) (2, 5) (2, 2, 5)\n",
      "[[1]\n",
      " [2]\n",
      " [3]\n",
      " [4]\n",
      " [5]] [[ 1  2]\n",
      " [ 3  4]\n",
      " [ 5  6]\n",
      " [ 7  8]\n",
      " [ 9 10]] [[[ 1  2]\n",
      "  [ 3  4]]\n",
      "\n",
      " [[ 5  6]\n",
      "  [ 7  8]]\n",
      "\n",
      " [[ 9 10]\n",
      "  [ 1  2]]\n",
      "\n",
      " [[ 3  4]\n",
      "  [ 5  6]]\n",
      "\n",
      " [[ 7  8]\n",
      "  [ 9 10]]]\n",
      "(5, 1) (5, 2) (5, 2, 2)\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a=np.array([1,2,3,4,5])\n",
    "b=np.array([[1,2,3,4,5],[6,7,8,9,10]])\n",
    "c=np.array([[[1,2,3,4,5],[6,7,8,9,10]],[[1,2,3,4,5],[6,7,8,9,10]]])\n",
    "print(a.ndim,b.ndim,c.ndim)\n",
    "print(type(a),type(b),type(c))\n",
    "print(np.shape(a),np.shape(b),np.shape(c))\n",
    "a1=np.reshape(a,(5,1))\n",
    "b1=np.reshape(b,(5,2))\n",
    "c1=np.reshape(c,(5,2,2))\n",
    "print(a1,b1,c1)\n",
    "print(np.shape(a1),np.shape(b1),np.shape(c1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "qioxjKxjPCVq",
    "outputId": "90373822-0bea-4999-b2bd-db776b9e5ebc"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 0 0]\n",
      " [0 0 0]]\n",
      "[[1 1 1]\n",
      " [1 1 1]]\n",
      "[[50. 50. 50.]\n",
      " [50. 50. 50.]]\n",
      "[[1 0 0 0]\n",
      " [0 1 0 0]\n",
      " [0 0 1 0]\n",
      " [0 0 0 1]]\n"
     ]
    }
   ],
   "source": [
    "z=np.zeros((2,3),int)\n",
    "o=np.ones((2,3),int)\n",
    "f=np.full((2,3),50,float)\n",
    "e=np.eye(4,dtype='int')\n",
    "print(z)\n",
    "print(o)\n",
    "print(f)\n",
    "print(e)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "WiYoYMc4QWCy",
    "outputId": "1ea838b0-9abf-4b36-8ba3-a2cdc151b390"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1  2  3  4  5]\n",
      " [ 6  7  8  9 10]]\n",
      "[[ 1  2  3  4  5]\n",
      " [ 6  7  8  9 10]]\n",
      "9\n",
      "9\n",
      "9\n",
      "[6 7]\n",
      "[7 8 9]\n",
      "[10  9  8  7  6]\n",
      "[[[1 2]\n",
      "  [6 7]]\n",
      "\n",
      " [[1 2]\n",
      "  [6 7]]]\n",
      "[[[ 1  2  3  4  5]\n",
      "  [ 6  7  8  9 10]]\n",
      "\n",
      " [[ 1  2  3  4  5]\n",
      "  [ 6  7  8  9 10]]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a=np.array([1,2,3,4,5])\n",
    "b=np.array([[1,2,3,4,5],[6,7,8,9,10]])\n",
    "c=np.array([[[1,2,3,4,5],[6,7,8,9,10]],[[1,2,3,4,5],[6,7,8,9,10]]])\n",
    "print(c[0])\n",
    "print(c[-1])\n",
    "print(b[1,3])\n",
    "print(b[-1,-2])\n",
    "print(c[1,1,-2])\n",
    "print(c[1,1,0:2])\n",
    "print(b[1,1:4])\n",
    "print(c[1,1,-1::-1])\n",
    "print(c[0:2,0:2,0:2])\n",
    "print(c[0:2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 7, 3, 4, 2],\n",
       "       [6, 8, 8, 9, 5]])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#sort axis=0: column sort, axis=1: row sort, \n",
    "import numpy as np\n",
    "a=np.array([[6,7,8,4,5],[1,8,3,9,2]])\n",
    "np.sort(a,axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[4, 5, 6, 7, 8],\n",
       "       [1, 2, 3, 8, 9]])"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a=np.array([[6,7,8,4,5],[1,8,3,9,2]])\n",
    "np.sort(a,axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[4, 5, 6, 7, 8],\n",
       "       [1, 2, 3, 8, 9]])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a=np.array([[6,7,8,4,5],[1,8,3,9,2]])\n",
    "np.sort(a,axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([(b'def', 80.3), (b'abc', 90.3), (b'bcd', 99. )],\n",
       "      dtype=[('name', 'S10'), ('perc', '<f8')])"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "d=np.dtype([('name','S10'),('perc',float)])\n",
    "student=np.array([('abc',90.3),('def',80.3),('bcd',99.0)],dtype=d)\n",
    "np.sort(student,order=\"perc\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 3 4 5] [ 1 10  3  4  5]\n"
     ]
    }
   ],
   "source": [
    "a=np.array([1,2,3,4,5])\n",
    "c=a.copy()\n",
    "c[1]=10\n",
    "print(a,c)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 1 10  3  4  5] [ 1 10  3  4  5]\n"
     ]
    }
   ],
   "source": [
    "a=np.array([1,2,3,4,5])\n",
    "v=a.view()\n",
    "v[1]=10\n",
    "print(a,v)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2]\n",
      " [3 4 5]]\n",
      "[[ 7  8  9]\n",
      " [10 11 12]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[ 0,  1,  2,  7,  8,  9],\n",
       "       [ 3,  4,  5, 10, 11, 12]])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a=np.arange(0,6).reshape(2,3)\n",
    "print(a)\n",
    "b=np.arange(7,13).reshape(2,3)\n",
    "print(b)\n",
    "np.concatenate((a,b),axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 6  7  8  4  5]\n",
      " [ 1  8  3  9  2]\n",
      " [ 1  2  3  4  5]\n",
      " [ 6  7  8  9 10]]\n",
      "[[ 6  7  8  4  5  1  2  3  4  5]\n",
      " [ 1  8  3  9  2  6  7  8  9 10]]\n"
     ]
    }
   ],
   "source": [
    "x=np.vstack((a,b))\n",
    "y=np.hstack((a,b))\n",
    "print(x)\n",
    "print(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[10 20 30 40 50 60 70 80 90]\n",
      "(array([2], dtype=int64),)\n",
      "(array([2, 5, 8], dtype=int64),)\n",
      "2\n"
     ]
    }
   ],
   "source": [
    "#searching\n",
    "a=np.arange(10,100,10)\n",
    "print(a)\n",
    "x=np.where(a==30)\n",
    "print(x)\n",
    "y=np.where(a%30==0)\n",
    "print(y)\n",
    "z=np.searchsorted(a,25)\n",
    "print(z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1]\n",
      " [2 3]]\n",
      "[1 2]\n",
      "[[1 3]\n",
      " [3 5]]\n",
      "[[-1 -1]\n",
      " [ 1  1]]\n",
      "[[5 6]\n",
      " [7 8]]\n",
      "[[0 2]\n",
      " [2 6]]\n",
      "[[0 1]\n",
      " [4 9]]\n"
     ]
    }
   ],
   "source": [
    "#arithmatic operations\n",
    "a=np.arange(0,4).reshape(2,2)\n",
    "print(a)\n",
    "b=np.array([1,2])\n",
    "print(b)\n",
    "c=np.array(5)\n",
    "a1=np.add(a,b)\n",
    "print(a1)\n",
    "b1=np.subtract(a,b)\n",
    "print(b1)\n",
    "c1=np.add(a,c)\n",
    "print(c1)\n",
    "d1=np.multiply(a,b)\n",
    "print(d1)\n",
    "e1=np.power(a,2)\n",
    "print(e1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[50 10 40]\n",
      " [30 60 20]\n",
      " [90 70 80]]\n"
     ]
    }
   ],
   "source": [
    "#statestical functions\n",
    "n=np.array([50,10,40,30,60,20,90,70,80])\n",
    "a=np.reshape(n,(3,3))\n",
    "print(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[30 10 20]\n",
      "[50 60 90]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "25.81988897471611"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.min(a)\n",
    "x=np.min(a,axis=0)\n",
    "print(x)\n",
    "np.max(a)\n",
    "y=np.max(a,axis=1)\n",
    "print(y)\n",
    "np.average(a)\n",
    "np.mean(a)\n",
    "np.mean(a,axis=0)\n",
    "np.median(a)\n",
    "np.median(a,axis=0)\n",
    "np.var(a)\n",
    "np.std(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "xpoints = np.array([1, 8])\n",
    "ypoints = np.array([3, 10])\n",
    "\n",
    "plt.plot(xpoints, ypoints)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "xpoints = np.array([1, 8])\n",
    "ypoints = np.array([3, 10])\n",
    "\n",
    "plt.plot(xpoints, ypoints, 'o')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXAAAAD4CAYAAAD1jb0+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAAqxElEQVR4nO3dd3RV15328e9W7wUkhBoITO9FCCFhO4ljx3EHTLWxKTY2djJOmWQyM5k3vUzapALGIDCO6Yg47iZuWBIghOi9XKECKiChiurd7x/IMw4GoXKlfc+9v89aLITKOQ8s9Ojcs8tRWmuEEEJYj4fpAEIIITpHClwIISxKClwIISxKClwIISxKClwIISzKqydPFhERoRMSEnrylEIIYXn79u27pLWOvP79PVrgCQkJ5OTk9OQphRDC8pRS52/0frmFIoQQFiUFLoQQFiUFLoQQFiUFLoQQFiUFLoQQFnXLAldKpSmlSpVSRz7zvl5KqR1KqdOtv4d3b0whhBDXa88V+Frg3uve9z3gfa31YOD91j8LIYToQbcscK31TqD8unc/DLzc+vbLwCOOjSWEEK7hamMLP3r9KAXldQ4/dmfvgUdprS8CtP7e52afqJRaopTKUUrllJWVdfJ0QghhTen7C1mTmcfFynqHH7vbBzG11iu11ola68TIyM+tBBVCCJdlt2vSMmyMjg1lUoLjhwo7W+AlSqlogNbfSx0XSQghXMPO02WcLatl8dQBKKUcfvzOFvjfgSdb334SeM0xcYQQwnWszrDRJ9iX+0ZHd8vx2zONcAOwCxiqlCpUSi0GfgncrZQ6Ddzd+mchhBCtTpVU88npSzyZkoCPV/fcrb7lboRa67k3+dBdDs4ihBAuY02mDV8vD+Ym9eu2c8hKTCGEcLDy2kbSc4uYPiGOXoE+3XYeKXAhhHCwV3efp6HZzqLUhG49jxS4EEI4UGOznXW7z3PHkEgGRwV367mkwIUQwoHeOHSBsuoGFk8d0O3nkgIXQggH0VqzOsPGoD5B3DE4otvPJwUuhBAOkm0r5+iFKhalds/CnetJgQshhIOszrARHuDN9AmxPXI+KXAhhHCA/Mt17DhewrzJ/fDz9uyRc0qBCyGEA6zJsuHloXhiSkKPnVMKXAghuqi6voktOYU8MCaGqBC/HjuvFLgQQnTRpr0F1DQ0syi1+6cOfpYUuBBCdEGLXbM2K4+khF6Mjgvt0XNLgQshRBfsOFZMYcVVFk1N6PFzS4ELIUQXrM6wEd/Ln7tH9O3xc0uBCyFEJx0qvMLevAoWpAzA06P7F+5cTwpcCCE6KS3DRpCvF7MS44ycXwpcCCE6obiynjcOXWRWYjzBft5GMkiBCyFEJ7yyOw+71izs5j2/2yIFLoQQHXS1sYVX9+Rz94go4nsFGMshBS6EEB2Uvr+QK3VNLJ460GgOKXAhhOgAu12TlmFjVGwIkxLCjWaRAhdCiA7YebqMs2W1LJ7aM3t+t0UKXAghOmB1ho0+wb7cPzrGdBQpcCGEaK9TJdV8cvoST0zpj4+X+fo0n0AIISxiTaYNXy8P5k3ubzoKIAUuhBDtUl7bSHpuEdMnxNEr0Md0HEAKXAgh2mX9nvM0NNtZZHDhzvWkwIUQ4hYam+2s23WeO4ZEMjgq2HSc/yUFLoQQt/Dm4QuUVjc41dU3SIELIUSbtNaszrAxqE8Qdw6JNB3nn0iBCyFEG7Jt5RwpqmJRqvmFO9eTAhdCiDakZdoIC/Bm2vhY01E+RwpcCCFuIv9yHe8dK+Gxyf3w9/E0HedzpMCFEOIm1mbl4akU85MTTEe5ISlwIYS4ger6JjbnFPDAmGj6hvqZjnNDXSpwpdQ3lVJHlVJHlFIblFLO+bcUQogO2rS3gJqGZuN7frel0wWulIoF/gVI1FqPAjyBOY4KJoQQprTYNWuz8piUEM7ouFDTcW6qq7dQvAB/pZQXEABc6Hoka2uxa9Zk2jhXVmM6ihCik3YcK6aw4iqLpw4wHaVNnS5wrXUR8BsgH7gIVGqt37v+85RSS5RSOUqpnLKyss4ntYidp8r40evHmL48i5y8ctNxhBCdsDrDRly4P3eP6Gs6Spu6cgslHHgYGADEAIFKqcev/zyt9UqtdaLWOjEy0rlWMXWH9dn59A70ITzAh8dW7eGdIxdNRxJCdMChwivszatgQUoCnh7OtXDnel25hfJlwKa1LtNaNwHpQIpjYllTaVU9H5wo5dHEOLYtTWFETAhLX81lbabNdDQhRDulZdgI8vVi9qR401FuqSsFng8kK6UC1LX1pXcBxx0Ty5q27Cukxa6ZM6kfvQJ9WP9UMncPj+KHrx/jF28dx27XpiMKIdpQUlXPG4cuMjMxjmA/b9Nxbqkr98D3AFuBXOBw67FWOiiX5djtmo1785kysDcDIgIB8PfxZPnjE5mf3J8Xd57jhU0HaGhuMZxUCHEz63bl0aI1C1Oce/DyU15d+WKt9Q+AHzgoi6Vlnr1EQflVvvOVYf/0fk8PxY8fHklsuD+/fPsEpVX1rHwikVB/5//pLoQ7udrYwqt78rlnRBT9egeYjtMushLTQTZmFxAe4M1XRkZ97mNKKZ698zb+MGccufkVzFyRxYUrVw2kFELczPb9RVypa2JRqjWuvkEK3CEu1TTw3rFipk+Iw9fr5hvePDwulpcXJnHxSj3TlmVy/GJVD6YUQtyM1pq0TBujYkNIGtDLdJx2kwJ3gG37Cmlq0cxNuvWodcqgCLYsnYJCMWvFLjLPXOqBhEKItuw8fYkzpTUsnup8e363RQq8i7TWbNxbwKSEcAb1ad+z8ob1DWH78ynEhPmzYE022/cXdnNKIURbVmfY6BPsy/2jY0xH6RAp8C7afa4c26Va5kzq16Gviw71Z/OzU0js34tvbjrIso/OoLVMMxSip50uqWbnqTKemNIfHy9rVaK10jqhjXvzCfHz4v4x0R3+2lB/b9YumsRDY2P41Tsn+a/XjtAic8WF6FFpmXn4enkwb3J/01E6rEvTCN1dRW0jbx8uZm5SPH7enXtah6+XJ7+fPY6YMH9WfHyW4soG/jR3vFM+/UMIV1Ne20h6biHTJ8TSK9DHdJwOkyvwLkjfX0Rji505SR27fXI9Dw/F9746jB8/PJL3T5Qw96XdXK5pcFBKIcTNrN9znoZmu6WmDn6WFHgnaa3ZmJ3PuPgwhkeHOOSYT0xJYPljEzl+sYoZy7M4f7nWIccVQnxeY7OddbvOc/vgCAZHtW8CgrORAu+kfecrOF1a066pgx1x76i+rH86mcqrTUxflsWBgisOPb4Q4po3D1+gtLrB6ff8bosUeCdtyC4gyNeLB8Y4ftrRxP7hbFuaQoCvJ3NX7ub94yUOP4cQ7kxrzeoMG7dFBnLHYOtucy0F3gmVV5t48/AFHhoXQ6Bv94wDD4wMIn1pKoP6BPH0uhzW78nvlvMI4Y725lVwpKiKRVMH4OHke363RQq8E147UER9k525HZz73VGRwb5sXJLMnUMi+Y/th/nteydlrrgQDrA64xxhAd5MHx9nOkqXSIF3kNaaDdkFjIoN6ZGHnQb6evHSE4nMmRTPnz44w79uOURTi73bzyuEq8q/XMd7x0qYl9TP8tN1pcA76FBhJccvVnV45WVXeHl68Ivpo/nW3UPYllvIorV7qa5v6rHzC+FK1mbl4akUT0xJMB2ly6TAO2hDdj7+3p48PK5n90xQSvEvdw3mV4+OIevsZWa/uJuSqvoezSCE1VXXN7E5p4AHxkTTN9TPdJwukwLvgJqGZv5+8AIPjo029rilWYnxpC2YxPnLtUxflsXpkmojOYSwos05hdQ0NLPIwlMHP0sKvANeP3iBusaWLq+87Ko7h0Sy6ZkpNLbYmbE8i2xbudE8QlhBi12zNsvGpIRwxsSFmY7jEFLgHbAhO5+hUcGMjw8zHYVRsaGkL00hItiXx1ft4c1DF01HEsKp7ThWQkH5Vcsum78RKfB2OlJUyaHCSuYmxTvNhu/xvQJIX5rCmLhQvrYhl1WfnDMdSQinlZZhIy7cn3tG9jUdxWGkwNtp4958fL08mOZk80bDAnz461OTuXdkX3765nF+/Pox7LIlrRD/5HBhJdl55SxIScDTwgt3ricF3g51jc28tv8C94+OJjTA+Z4m7+ftyZ/nTWBhagJpmTa+tiGX+qYW07GEcBppmTYCfTyZNcmxexeZJgXeDm8eukh1Q7Pxwcu2eHoofvDgSL5//3DeOlzM/NV7uFLXaDqWEMaVVNXz+sELzJoUT4ih2WPdRQq8HTZk53NbZCCTEsJNR7mlp24fyJ/mjudgQSWPrthFYUWd6UhCGLVuVx4tWrMwxXUGLz8lBX4Lp0qqyc2/wtykfk4zeHkrD46N4ZXFSZRW1TNtWRZHiipNRxLCiKuNLazfk8/dw6Po1zvAdByHkwK/hQ3Z+fh4ejB9gnMNXt7K5IG92bo0BW8PxewXd7HzVJnpSEL0uO37i6ioa7L0nt9tkQJvQ31TC+m5RdwzMsqSz8sbEhXM9udT6dc7kEVr97Ilp8B0JCF6jNaatEwbI2NCSBrQy3ScbiEF3oZ3jhRTebWJeU48eHkrUSF+bH4mmeSBvfnO1kP86f3TsiWtcAs7T1/iTGkNi6cOsMztz46SAm/Dhux8+vcOIHlgb9NRuiTYz5u0BZOYPj6W3+44xX9sP0yzbEkrXNzqDBuRwb7d8tQsZyEFfhNny2rYYytn9qR4Sz+x41M+Xh78dtZYvvbFQWzILmDJK/uoa2w2HUuIbnG6pJqdp8p4Irk/Pl6uW3Ou+zfrok17C/DyUDw60VqDl21RSvGvXxnKz6aN4qOTpcxZuZuy6gbTsYRwuLTMPHy9PJg32bq3P9tDCvwGGppb2LqvkC8Pj6JPsPX3DL7eY5P7s3J+IqdKqpmxPItzZTWmIwnhMOW1jaTnFjJtfCy9g3xNx+lWUuA3sONYCeW1jcxJcq1lt5/15RFRbFwyhZqGZmYszyI3v8J0JCEcYkN2Pg3NdpfZ87stUuA3sDG7gNgwf24fHGk6SrcaFx9G+tIUQvy9mbtyN+8dLTYdSYguaWy283JWHrcPjmBIVLDpON1OCvw65y/XknHmErMnxbvUrmU3kxARSPrSFIZFh/DsX/fxyq4805GE6LS3Dl+ktLrBLa6+oYsFrpQKU0ptVUqdUEodV0pNcVQwUzbtLcBDXXt0mbvoHeTLxqeT+dKwPvzXa0f55dsnZEtaYTlaa1Zn2LgtMpA7XfzV86e6egX+B+AdrfUwYCxwvOuRzGlqsbNlXyFfGtbHJR542hH+Pp6seHwij03ux4qPz/KtzQdobJa54sI69uZVcLiokkVTB7jE1N/28OrsFyqlQoA7gAUAWutGwNL7l75/vJSy6gbmTHLtqUc34+XpwU8fGUVMmD+/fvckpdUNrJg/0eW24BSuKS3DRliAN9Od7KEr3akrV+ADgTJgjVJqv1JqlVIq8PpPUkotUUrlKKVyysqce0OljXvz6RvixxeGusfLrxtRSvH8Fwfxu1ljybaVM2vFLi5WXjUdS4g2FZTX8d6xYuYl9cPfx9N0nB7TlQL3AiYAy7XW44Fa4HvXf5LWeqXWOlFrnRgZ6bzFWHTlKh+fKmNWYhxenjK2O31CHGsXJlFYcZXpy7I4WVxtOpIQN7U2Kw8PpXhiSoLpKD2qK01VCBRqrfe0/nkr1wrdkjbtvbZTn6s9cqkrpg6OYPMzU7BrzaMrssg6e8l0JCE+p7q+iU17C7h/TLTbjV11usC11sVAgVJqaOu77gKOOSRVD2tusbMlp4A7BkcSF+56m753xYiYENKfS6VviB9PpmXz2oEi05GE+CebcwqpaWh22T2/29LVewVfB15VSh0CxgE/73IiAz4+VcbFynrmuvDKy66IDfNn67MpTOgXzgsbD/Dix2dlS1rhFFrsmrVZNhL7hzMmLsx0nB7XpQLXWh9ovb89Rmv9iNbakuuxN2QXEBHky13Do0xHcVqhAd6sW5zEA2Oi+cXbJ/jh34/SInPFhWE7jpVQUH7VLa++oQvTCF1FcWU9H54sZckdA/GWwcs2+Xp58sc544kJ82flznMUV9Xzhznj8fN2n1F/4VzSMmzEhftzz8i+pqMY4faNtSWngBa7Zo4MXraLh4fiP+4bzg8eHMF7x0qY99JuKmotPf1fWNThwkqy88pZkJLgFtte3IhbF7jdrtmUU0DqoN707/25KeyiDQtTB7Bs3gSOXKhixvIsCsrrTEcSbiYt00agj6dbzxxz6wLPOHOJwoqrbrvysqu+Ojqa9U9N5nJtI9OWZXKo8IrpSMJNlFTV88ahC8xMjHfrlcJuXeAbsvMJD/DmnpEyeNlZiQm92LY0BV8vT+as3M2HJ0tNRxJu4JVd52m2axamJpiOYpTbFnhZdQM7jpXw6MQ4fL1kEK4rBvUJYvvzKQyMDOSpl3PYtDffdCThwuqbWnh1z3nuHh7l9rc+3bbAt+UW0mzXzJbbJw7RJ9iPjUumkDoogn/bdpj/2XFK5oqLbrF9fxEVdU1us+d3W9yywLXWbMzOJymhF4P6BJmO4zKCfL1Y/WQiMyfG8Yf3T/PdrYdoapEtaYXjaK1Jy7AxMiaEyQN6mY5jnFsW+K5zl8m7XMfcye47et1dvD09+NWjY3jhrsFs2VfI4pdzqGloNh1LuIidpy9xurSGRakDUMo9pw5+llsW+IbsAkL8vPjqqGjTUVySUopv3j2E/54xmswzl5izchel1fWmYwkXkJZhIzLYlwfHxpiO4hTcrsDLaxt590gx0yfEyQrCbjZ7Uj9WPZnIubJapi/L4kxpjelIwsLOlFbz8akynkjuj4+X21XXDbndv0J6biGNLXbmyMZVPeKLQ/uwcUky9U0tPLoii5y8ctORhEWlZebh4+XBvMky8eBTblXgWms2ZOczvl8Yw/qGmI7jNsbEhZG+NJVeAT7MW7WHtw9fNB1JWExFbSPpuYVMHx9L7yBf03GchlsVeM75Cs6W1TJXpg72uH69A9i6NIVRMSE8tz6XNZk205GEhazPzqe+yS5TB6/jVgW+ITufIF8vHhgrg5cm9Ar0Yf3TydwzIoofvX6Mn715DLtsSStuobHZzrpdedw+OIIhUcGm4zgVtynwyrom3jx0kYfHxRDg4/a76Brj5+3Jsscm8uSU/rz0iY1/2bifhuYW07GEE3vr8EVKqhrk6vsG3KbJ/nagiIZmO3OT5PaJaZ4eih8+NJKYMH9+8fYJyqobWDk/kdAA992USNyY1prVGTYGRgZy52DnfSi6KW5xBf7p4OXo2FBGxYaajiO4Nlf8mTtv4w9zxpGbX8HMF7MounLVdCzhZHLOV3C4qJJFqQPwcNM9v9viFgV+oOAKJ4qrZeqgE3p4XCwvL0riYmU905dlcuxClelIwoms/sRGqL83MybEmY7ilNyiwDdmFxDg48lDsnrLKaXcFsGWZ6fgoRSzXtxFxulLpiMJJ1BQXsd7x4qZN7kf/j6y6O5GXL7Aq+ubeP3QBR4cE0OwG2/87uyG9Q0h/bkU4sL9WbAmm/TcQtORhGFrs/LwUIonpySYjuK0XL7A/37wAnWNLXL7xAKiQ/3Z/OwUkgb04lubD/KXD8/IlrRuqrq+iU17C7h/TDR9Q/1Mx3FaLl/gG7MLGNY3mHHxYaajiHYI8fNm7cIkHhkXw6/fPcn3/3aEZtmS1u1sySmkpqGZRakydbAtLj2N8EhRJYeLKvnRQyNl60kL8fHy4HezxhEd5s/yj85SUlXPH+eOl/n7bqLFrlmTZSOxfzhj5cKrTS59Bb4hOx9fLw8eGRdrOoroIA8Pxb/dO4yfPDySD06UMvelPVyuaTAdS/SAHcdKKCi/Kgt32sFlC7yusZnXDlzg/jHRskDEwuZPSWDF4xM5WVzFjOVZ5F2qNR1JdLO0TBuxYf7cM0IeNn4rLlvgbxy8SE1Ds6y8dAH3jOzL+qeTqbzaxIzlWRwouGI6kugmR4oqybaVszA1AS9Pl60nh3HZf6ENe/MZ1CeIxP7hpqMIB5jQL5xtS1MI9PVizspd/ONYielIohukZdgI9PFk1iSZNdYeLlngJ4qr2J9/hTmT4mXw0oUMjAxi29IUhkQFs+SVHF7dc950JOFApVX1vH7oAjMT4wmRNRvt4pIFvjG7AB9PD6bL8luXExnsy8YlyXxhaB/+c/sRfvPuSZkr7iLW7TpPs12zMDXBdBTLcLkCr29qIT23kHtH9aVXoI/pOKIbBPh4sXL+ROYmxfPnD8/w7S0HaWyWueJWVt/Uwqt7zvPl4VH07x1oOo5luNzE2rePXKSqvllWXro4L08Pfj5tNDGh/vx2xylKqxpY/vgE2S7BorbvL6KironFMnWwQ1zuCnzDngISegcwZWBv01FEN1NK8fW7BvPrR8ew+9xlZr24m5KqetOxRAdprUnLsDEiOoTJA3qZjmMpLlXgZ0pryM4rZ05SPxm8dCMzE+NJWzCJ/Mu1TPtLJqdLqk1HEh3wyelLnC6tYfHUAfJ920EuVeAbs/Px8lCyd7AbumNIJJuemUKTXTNjeRZ7zl02HUm00+oMG5HBvvKs2k7ocoErpTyVUvuVUm84IlBnNTS3sC23kLtHRBEZ7GsyijBkVGwo259LoU+IH/NXZ/PGoQumI4lbOFNazcenypif3B9fL9nzu6MccQX+AnDcAcfpkveOllBR1yQrL91cXHgAW5+dwrj4ML62fj+rPjlnOpJoQ1pmHj5eHjw2Wb5vO6NLBa6UigPuB1Y5Jk7nbcjOJy7cn6mDIkxHEYaFBfiwbnES943uy0/fPM6PXj9Ki13mijubitpG0nMLmTYult5B8qq5M7p6Bf574LvATSfhKqWWKKVylFI5ZWVlXTzdjZ2/XEvW2cvMmRQvDz4VAPh5e/LnuRNYlDqANZl5fG19LvVNLaZjic9Yn51PfZNddh3sgk4XuFLqAaBUa72vrc/TWq/UWidqrRMjIyM7e7o2bdxbgKeHYmaizP0W/8fDQ/H/HhzB9+8fzjtHi5m/eg9X6hpNxxJAY7OddbvyuH1wBEP7BpuOY1lduQJPBR5SSuUBG4EvKaX+6pBUHdDUYmdLTiFfHNqHqBB59JL4vKduH8if507gYGElM5ZnUVBeZzqS23v7yEVKqhrkiTtd1OkC11r/u9Y6TmudAMwBPtBaP+6wZO30/vESLtU0MG+yXH2Lm7t/TDR/XTyZsuoGpi/P4khRpelIbktrzeoMGwMjA7lzSPe8KncXlp8HviG7gOhQP+4c0sd0FOHkkgb0YtvSFHw8PZj94i4+PtU9YzKibTnnKzhUWMnC1AEyZtVFDilwrfVHWusHHHGsjigor2Pn6TJmJsbjKf8RRDsMjgom/bkU+vUOZNHavWzJKTAdye2kZdgI9fdmxgR51GFXWfoK/NNvvtmy+bvogKgQPzY/k0zKbb35ztZD/PH907IlbQ8pKK/j3aPFzJvcTx5S7QCWLfDmFjubcgq4c0gksWH+puMIiwn28yZtwSRmTIjjdztO8e/ph2lukS1pu9varDw8lOKJKf1NR3EJlv0R+NHJMkqqGvjRQ7KCS3SOt6cHv5k5hpgwP/70wRlKqur587wJBPpa9tvCqVXXN7FpbwH3jY4mOlQuuhzBslfgG/fmExnsy13DZfBSdJ5Sim/fM5SfTxvNx6fKmPvSbsqqG0zHcklbcgqpaWiWhTsOZMkCv1h5lQ9OlDJzYhze8uRq4QDzJvfjpScSOV1Sw/TlmZwrqzEdyaW02DVrs/KY2D+ccfFhpuO4DEu235acQuxaBi+FY901PIoNS5Kpa2hhxvIs9p2vMB3JZfzjeAn55XXyxB0Hs1yBt9g1m/YWMHVQhDw7TzjcuPgw0p9LIdTfm3kv7ebdo8WmI7mE1Rk2YsP8uWdElOkoLsVyBf7J6TKKrlyVZ16KbtO/dyDblqYwPDqEZ/+6j3W78kxHsrQjRZVk28pZkJKAl9zydCjL/WtuzC6gd6AP94zoazqKcGG9g3zZ8HQydw2L4v+9dpRfvn0Cu2xJ2ylpGTYCfTyZLRddDmepAi+trucfx0uYMTEOHy9LRRcW5O/jyYvzJ/J4cj9WfHyWb24+QEOzbEnbEaVV9bx+6AIzE+MJ8fM2HcflWGrC69Z9hTTbtQxeih7j6aH4ycOjiAnz51fvnKS0qoEV8ycS6i9l1B6v7D5Ps12zMDXBdBSXZJnLWHvr4OXkAb24LTLIdBzhRpRSPPeFQfzP7LHknC9n1opdXLhy1XQsp1ff1MKre/L58vAomXDQTSxT4LvPXeb85Tp55qUwZtr4ONYuTOLClatMX5bFieIq05Gc2t/2F1Fe2yh7fncjyxT4+ux8Qv29uXeUDF4Kc1IHRbD52SkAzFy+i6wzlwwnck5aa9IybYyIDiF5YC/TcVyWJQr8ck0D7x0tYfqEWPy8PU3HEW5ueHQI6c+lEB3mx5NrsnntQJHpSE7nk9OXOFVSw6KpA1BKtnruLpYo8PTcIhpb7HL7RDiNmDB/tjybwsT+4byw8QArPj4rW9J+RlqmjYggXx4cG206ikuzRIF7eyruGRHFkCh5+KlwHqH+3ry8KIkHx8bwy7dP8IO/H6VF5opzprSaj06W8cSU/vh6ySvm7mSJaYQLUgewQAZChBPy9fLkD7PHERPqx4s7z1FcWc8f545361t9aZl5+Hh5MG+yvGLubpa4AhfCmXl4KP79vuH88MER7DhewryXdlNe22g6lhEVtY2k5xYybVwsEUG+puO4PClwIRxkQeoAlj82gaMXqpixPIv8y3WmI/W49dn51DfZWTg1wXQUtyAFLoQD3TsqmlefmkxFXSPTl2dyqPCK6Ug9pqnFzrpdeUwdFMGwviGm47gFKXAhHCwxoRfblqbg5+3J7Bd38+GJUtOResRbhy9SUtUge373IClwIbrBbZFBpD+XwqA+QTy1LoeN2fmmI3UrrTWrM2wMjAzkziGRpuO4DSlwIbpJn2A/Ni5JZuqgCL6Xfpjf7TjlsnPF952v4FBhJQtTB+DhIQt3eooUuBDdKNDXi1VPJjIrMY4/vn+a72w9RFOL3XQsh1udYSPU35sZE2JNR3ErlpgHLoSVeXt68N8zxhAT5s/v/3Ga0uoGlj02gSBf1/j2Kyiv492jxSy54zYCfFzj72QVcgUuRA9QSvGNLw/hVzPGkHnmErNf3EVpVb3pWA7xclYeSimeTOlvOorbkQIXogfNmhTP6icTsV2qZdqyLM6UVpuO1CU1Dc1s2lvAfaOjiQ71Nx3H7UiBC9HDvjC0D5uWTKGh2c6M5bvYm1duOlKnbckpoLqhWaYOGiIFLoQBo+NC2f5cCr2DfHhs1R7ePnzRdKQOa7Fr1mTmMbF/OOPiw0zHcUtS4EIYEt8rgG3PpjA6NpTn1ueSlmEzHalD/nG8hPzyOnnijkFS4EIYFB7ow6tPTeYrI/ry4zeO8dM3jmG3yJa0aRk2YsP8+crIKNNR3JYUuBCG+Xl78pfHJrAgJYFVGTa+vnE/9U0tpmO16UhRJXts5SxIScDLU2rEFJm0KYQT8PRQ/ODBEcSG+fOzt45TVt3AS/MTCQ3wNh3thtIybQT4eDJrUrzpKG5NfnQK4SSUUjx9x0D+OHc8B/KvMGNFFkVXrpqO9TmlVfW8fvACsxLjCfV3zh8w7qLTBa6UildKfaiUOq6UOqqUesGRwYRwVw+NjeHlRUmUVNUz7S+ZHL1QaTrSP3ll93ma7ZoFKQmmo7i9rlyBNwPf1loPB5KB55VSIxwTSwj3NuW23mxbmoKXh2L2i7v55HSZ6UgA1De18OqefO4aFkVCRKDpOG6v0wWutb6otc5tfbsaOA7ITjZCOMiQqGDSn0slLtyfhWv2kp5baDoSf9tfRHltoyzccRIOuQeulEoAxgN7bvCxJUqpHKVUTlmZc1xFCGEVfUP92PzsFCYP7MW3Nh/kLx+eMbYlrdaatEwbw6NDSB7Yy0gG8c+6XOBKqSBgG/ANrXXV9R/XWq/UWidqrRMjI2WjdyE6KsTPmzULkpg2PpZfv3uS//zbEZoNbEmbceYSp0pqWDx1AErJnt/OoEvTCJVS3lwr71e11umOiSSEuJ6Plwe/mzWW6FA/ln10ltKqev44d3yPbt+6OsNGRJAvD46N7rFzirZ1ZRaKAlYDx7XWv3NcJCHEjSil+O69w/jJI6P44EQpc1/aw6Wahh4595nSGj46Wcb85P74enn2yDnFrXXlFkoqMB/4klLqQOuv+xyUSwhxE/OT+/Pi/EROFlcxY3kWeZdqu/2cazJt+Hh58Fhyv24/l2i/rsxCydBaK631GK31uNZfbzkynBDixu4eEcX6p5Oprm9m+vIs9udXdNu5Kmob2ZZbyCPjYogI8u2284iOk5WYQljUhH7hbFuaQrCfF3Nf2s2OYyXdcp4Ne/Opb7KzSKYOOh0pcCEsbEBEINuWpjA0KphnXsnhr7vPO/T4TS121mWdZ+qgCIb1DXHosUXXSYELYXERQb5sWJLMF4f24ft/O8Kv3jnhsLnibx2+SHFVPYumJjjkeMKxpMCFcAEBPl68OH8i8yb3Y9lHZ/n25oM0NndtrrjWmtUZNgZGBPKFIX0clFQ4kmwnK4SL8PL04GePjCI2zJ9fv3uS0uoGlj8+gWC/zu0YuO98BYcKK/nJwyPx8JCFO85IrsCFcCFKKZ7/4iB+O3Msu89dZuaKXRRX1nfqWGmZNkL9vZkxMc7BKYWjSIEL4YJmTIxjzcJJFFZcZfqyTE6VVHfo6wvK63jnSDFzk/r16GpP0TFS4EK4qNsHR7LpmWSa7ZoZy7PYfe5yu7/25aw8lFI8MaV/NyYUXSUFLoQLGxkTyvbnU4kK8eOJ1dm8fvDCLb+mpqGZTXsLuG90NDFh/j2QUnSWFLgQLi42zJ9tz6Ywrl8YX9+wn5d2nmtzmuGWnAKqG5plz28LkAIXwg2EBnizblES94+J5mdvHefHbxyjxf75Em+xa9Zk5jGhXxjj4sN6PqjoEBmdEMJN+Hl78qc544kO8WNVho3iynr+Z/Y4/Lz/b3fB94+XkF9ex7/dO8xgUtFecgUuhBvx8FB8/4ER/NcDI3jnaDGPr9pDRW3j/358dYaN2DB/vjIyymBK0V5S4EK4ocVTB/CXeRM4VFTJjBVZFJTXcaSokj22cp5M6Y+Xp1SDFcgtFCHc1H2jo4kI8uXpdTlMW5bF4D5BBPh4MnuS7PltFfJjVgg3ljSgF9uWTsHXy4Nd5y4zc2Icof6dW3ovep5cgQvh5gb1CWb7cym89Mk5nr5joOk4ogOkwIUQ9Anx4z/vH2E6hugguYUihBAWJQUuhBAWJQUuhBAWJQUuhBAWJQUuhBAWJQUuhBAWJQUuhBAWJQUuhBAWpdra2N3hJ1OqDDjfyS+PAC45ME53s1JeK2UFa+W1UlawVl4rZYWu5e2vtY68/p09WuBdoZTK0Vonms7RXlbKa6WsYK28VsoK1sprpazQPXnlFooQQliUFLgQQliUlQp8pekAHWSlvFbKCtbKa6WsYK28VsoK3ZDXMvfAhRBC/DMrXYELIYT4DClwIYSwKKcvcKVUmlKqVCl1xHSWW1FKxSulPlRKHVdKHVVKvWA6U1uUUn5KqWyl1MHWvD8ynelWlFKeSqn9Sqk3TGe5FaVUnlLqsFLqgFIqx3SetiilwpRSW5VSJ1r//04xnelmlFJDW/9NP/1VpZT6hulcN6OU+mbr99cRpdQGpZSfw47t7PfAlVJ3ADXAOq31KNN52qKUigaitda5SqlgYB/wiNb6mOFoN6SUUkCg1rpGKeUNZAAvaK13G452U0qpbwGJQIjW+gHTedqilMoDErXWTr/YRCn1MvCJ1nqVUsoHCNBaXzEc65aUUp5AETBZa93ZRYLdRikVy7XvqxFa66tKqc3AW1rrtY44vtNfgWutdwLlpnO0h9b6otY6t/XtauA4EGs21c3pa2pa/+jd+stpf6IrpeKA+4FVprO4EqVUCHAHsBpAa91ohfJudRdw1hnL+zO8AH+llBcQAFxw1IGdvsCtSimVAIwH9hiO0qbWWxIHgFJgh9bamfP+HvguYDeco7008J5Sap9SaonpMG0YCJQBa1pvT61SSgWaDtVOc4ANpkPcjNa6CPgNkA9cBCq11u856vhS4N1AKRUEbAO+obWuMp2nLVrrFq31OCAOSFJKOeVtKqXUA0Cp1nqf6SwdkKq1ngB8FXi+9XagM/ICJgDLtdbjgVrge2Yj3VrrrZ6HgC2ms9yMUioceBgYAMQAgUqpxx11fClwB2u9l7wNeFVrnW46T3u1vmT+CLjXbJKbSgUear2vvBH4klLqr2YjtU1rfaH191JgO5BkNtFNFQKFn3n1tZVrhe7svgrkaq1LTAdpw5cBm9a6TGvdBKQDKY46uBS4A7UOCq4Gjmutf2c6z60opSKVUmGtb/tz7T/bCaOhbkJr/e9a6zitdQLXXjZ/oLV22JWMoymlAlsHsmm9HXEP4JQzqbTWxUCBUmpo67vuApxy4P06c3Hi2yet8oFkpVRAaz/cxbWxMYdw+gJXSm0AdgFDlVKFSqnFpjO1IRWYz7Wrw0+nON1nOlQbooEPlVKHgL1cuwfu9NPzLCIKyFBKHQSygTe11u8YztSWrwOvtv5fGAf83GyctimlAoC7uXZF67RaX9VsBXKBw1zrXIctqXf6aYRCCCFuzOmvwIUQQtyYFLgQQliUFLgQQliUFLgQQliUFLgQQliUFLgQQliUFLgQQljU/wdir8VNt1uw4QAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "xpoints = np.array([1, 2, 6, 8])\n",
    "ypoints = np.array([3, 8, 1, 10])\n",
    "\n",
    "plt.plot(xpoints, ypoints)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "ypoints = np.array([3, 8, 1, 10, 5, 7])\n",
    "\n",
    "plt.plot(ypoints)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "ypoints = np.array([3, 8, 1, 10])\n",
    "\n",
    "plt.plot(ypoints, marker = 'o')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 265
    },
    "id": "jZ6YIlafC0KN",
    "outputId": "59bb2032-dfdd-40e1-b65d-88eaf73985ed"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "x = np.array([0, 1, 2, 3])\n",
    "y = np.array([3, 8, 1, 10])\n",
    "\n",
    "plt.subplot(1,2,1)\n",
    "plt.plot(x,y)\n",
    "\n",
    "\n",
    "#plot 2:\n",
    "x = np.array([0, 1, 2, 3])\n",
    "y = np.array([10, 20, 30, 40])\n",
    "\n",
    "plt.subplot(1,2, 2)\n",
    "plt.plot(x,y)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "ljZeH4eXG8PM"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#plot 2:\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "x = np.array([0, 1, 2, 3])\n",
    "y = np.array([10, 20, 30, 40])\n",
    "\n",
    "plt.subplot(1,2, 2)\n",
    "plt.plot(x,y)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {
    "id": "Ib_4KAw-GeM3"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3.738891832440735"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import scipy.stats as s\n",
    "s.f.ppf(1-0.05,2,14)\n"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "Untitled1.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
