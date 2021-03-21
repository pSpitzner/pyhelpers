# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-21 11:11:40
# @Last Modified: 2021-03-21 13:28:32
# ------------------------------------------------------------------------------ #
# Helper functions to work conveniently with hdf5 files
#
# Example
# ```
# import hi5 as h5
# fpath = '~/demo/file.hdf5'
# h5.ls(fpath)
# h5.recursive_ls(fpath)
# h5.load(fpath, '/some/dataset/')
# h5.recursive_load(fpath, hot=False)
# ```
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import numbers
import h5py
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)


def load(filenames, dsetname, keepdim=False, raise_ex=False, silent=False):
    """
        load a h5 dset into an array. opens the h5 file and closes it
        after reading.

        # Parameters
        filenames: str path to h5file(s).
                   if wildcard given, result from globed files is returned
        dsetname:  str, which dset to read
        keepdim:   bool, per default arrays with 1 element will be mapped to scalars
                   set this to `True` to keep them as arrays
        raise_ex: whether to raise exceptions. default false,
                  in this case, np.nan is returned if loading fails
        silent:   if set to true, exceptions will not be reported

        # Returns
        res: ndarray or scalar, depending on loaded datatype
    """

    def local_load(filename):
        try:
            file = h5py.File(filename, "r")
            res = file[dsetname]
            # map 1 element arrays to scalars
            if res.shape == (1,) and not keepdim:
                res = res[0]
            elif res.shape == ():
                res = res[()]
            else:
                res = res[:]
            file.close()
            return res
        except Exception as e:
            if not silent:
                log.error(f"failed to load {dsetname} from {filename}")
            if raise_ex:
                raise e
            else:
                return np.nan

    files = glob.glob(os.path.expanduser(filenames))
    res = []
    for f in files:
        res.append(local_load(f))

    if len(files) == 1:
        return res[0]
    else:
        return res


def ls(filename, dsetname="/"):
    """
        list the keys in a dsetname

        Parameters
        ----------
        filename: path to h5file
        dsetname: which dset to list

        Returns
        -------
        list: containing the contained keys as strings
    """
    try:
        file = h5py.File(os.path.expanduser(filename), "r")
        try:
            res = list(file[dsetname].keys())
        except Exception as e:
            res = []
        file.close()
    except Exception as e:
        res = []

    return res


_h5_files_currently_open = []


def load_hot(filename, dsetname, keepdim=False):
    """
        sometimes we do not want to hold the whole dataset in RAM, because it is too
        large. Remember to close the file after processing!
    """
    file = h5py.File(os.path.expanduser(filename), "r")
    global _h5_files_currently_open
    _h5_files_currently_open.append(file)

    # if its a single value, load it nonetheless
    if file[dsetname].shape == (1,) and not keepdim:
        return file[dsetname][0]
    elif file[dsetname].shape == ():
        return file[dsetname][()]
    else:
        return file[dsetname]


def close_hot(which="all"):
    """
        hot files require a bit of care:
        * If a BetterDict is opened from a hot hdf5 file, and `all` hot files are closed, the datasets are no longer accessible.
        * from the outside, it is hard to check whether an element is
    """
    global _h5_files_currently_open
    if which == "all":
        for file in _h5_files_currently_open:
            try:
                file.close()
            except:
                log.debug("File already closed")
        _h5_files_currently_open = []
    else:
        _h5_files_currently_open[which].close()
        del _h5_files_currently_open[which]


def recursive_ls(filename, dsetname=""):
    if dsetname == "":
        dsetname = "/"

    candidates = ls(filename, dsetname)
    res = candidates.copy()
    for c in candidates:
        temp = recursive_ls(filename, dsetname + f"{c}/")
        if len(temp) > 0:
            temp = [f"{c}/{el}" for el in temp]
            res += temp
    return res


def recursive_load(filename, dsetname="/", skip=None, hot=False):
    """
        Load a hdf5 file as a nested BetterDict.

        # Paramters:
        skip : list
            names of dsets to exclude
        hot : bool
            if True, does not load dsets to ram, but only links to the hdf5 file. this keeps the file open, call `close_hot()` when done!
            Use this if a dataset in your file is ... big
    """
    if skip is not None:
        assert isinstance(skip, list)
    else:
        skip = []

    candidates = recursive_ls(filename, dsetname)

    cd_len = []
    res = BetterDict()

    maxdepth = 0
    for cd in candidates:
        l = len(cd.split("/"))
        cd_len.append(l)
        if l > maxdepth:
            maxdepth = l

    # iterate by depth, creating hierarchy
    for ddx in range(1, maxdepth + 1):
        for ldx, l in enumerate(cd_len):
            if l == ddx:
                cd = candidates[ldx]
                components = cd.split("/")
                if len([x for x in skip if x in components]) > 0:
                    continue
                temp = res
                if ddx > 1:
                    for cp in components[0:-1]:
                        temp = temp[cp]
                cp = components[-1]
                if len(ls(filename, cd)) > 0:
                    temp[cp] = BetterDict()
                else:
                    if hot:
                        temp[cp] = load_hot(filename, cd)
                    else:
                        temp[cp] = load(filename, cd)

    return res


class BetterDict(dict):
    """
        Class for loaded hdf5 files --- a tweaked dict that supports nesting

        We inherit from dict and also provide keys as attributes, mapped to `.get()` of
        dict. This avoids the KeyError: if getting parameters via `.the_parname`, we
        return None when the param does not exist.
        Avoid using keys that have the same name as class functions etc.

        # Example
        ```
        >>> foo = BetterDict(lorem="ipsum")
        >>> print(foo.lorem)
        ipsum
        >>> print(foo.does_not_exist is None)
        True
        ```
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # copy everything
    def __deepcopy__(self, memo=None):
        return SimulationResult(copy.deepcopy(dict(self), memo=memo))

    @property
    def varnames(self):
        return [*self]

    @property
    def depth(self):
        maxdepth = 0
        for vdx, var in enumerate(self.varnames):
            if isinstance(self[var], BetterDict):
                d = self[var].depth
                if d > maxdepth:
                    maxdepth = d
        return maxdepth + 1

    # helper to call recursively and build the tree structure of the content
    def __recursive_tree__(self, d=None, depth=0):
        if d is None:
            d = {key: [] for key in ["pcs", "scs", "varname", "varval", "vartype"]}
            d["prev_pc"] = ""

        for vdx, var in enumerate(self.varnames):
            if vdx < len(self.varnames) - 1:
                sc = "├── "
                pc = "│   "
            else:
                sc = "└── "
                pc = "    "

            d["pcs"].append(d["prev_pc"])
            d["scs"].append(sc)
            d["varname"].append(var)

            if isinstance(self[var], BetterDict):
                # get all content of the dict, while incrementing the padding `pc`
                d["vartype"].append("")
                d["varval"].append("")
                d["prev_pc"] += pc
                self[var].__recursive_tree__(d, depth + 1)
                d["prev_pc"] = d["prev_pc"][0:-4]
            else:
                # extract type and certain variables
                d["vartype"].append(f"{self[var].__class__.__name__}")
                # number
                if isinstance(self[var], numbers.Number):
                    d["varval"].append(str(self[var]))
                # numpy byte strings
                elif isinstance(self[var], np.bytes_):
                    string = self[var].decode("UTF-8").replace("\n", " ")
                    if len(string) > 14:
                        string = f"{string:.11s}..."
                    d["varval"].append(string)
                # base strings
                elif isinstance(self[var], str):
                    string = self[var].replace("\n", " ")
                    if len(string) > 14:
                        string = f"{string:.11s}..."
                    d["varval"].append(string)
                # numpy arrays, print shape
                elif isinstance(self[var], np.ndarray):
                    d["varval"].append(f"{self[var].shape}")
                # list, print length
                elif isinstance(self[var], list):
                    d["varval"].append(f"({len(self[var])})")

                else:
                    d["varval"].append("")

        return d

    # printed representation
    def __repr__(self):
        res = ""
        d = self.__recursive_tree__()
        for l in range(0, len(d["varname"])):
            left = f"{d['pcs'][l]}{d['scs'][l]}{d['varname'][l]}"
            right = f"{d['vartype'][l]}"
            ws = " " if d["vartype"][l] is "" else "."
            res += f"{left} {ws*(62-len(left)-len(right))} {right}"
            res += f"  {d['varval'][l]}\n" if len(d["varval"][l]) > 0 else "\n"

        return res

    # def __repr__(self):
    #     res = ""
    #     for vdx, var in enumerate(self.varnames):
    #         if vdx == len(self.varnames) - 1:
    #             sc = "└── "
    #             pc = "     "
    #         else:
    #             sc = "├── "
    #             pc = "│    "
    #         if isinstance(self[var], BetterDict):
    #             temp = repr(self[var])
    #             temp = temp.replace("\n", f"\n{pc}", temp.count("\n") - 1)
    #             temp = f"{sc}{var}\n{pc}" + temp
    #         else:
    #             left = f"{sc}{var}"
    #             right = f"{self[var].__class__.__name__}"
    #             if isinstance(self[var], numbers.Number):
    #                 right = f"{self[var]} ({right})"
    #             temp = f"{left} {'.'*(70-len(left)-len(right))} {right}\n"
    #             # temp = f"{left} ({right})\n"
    #         res += temp
    #     return res

    # enable autocompletion. man this is beautiful!
    def __dir__(self):
        res = dir(type(self)) + list(self.varnames)
        return res
