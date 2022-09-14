
import numpy as np
import os

from datetime import datetime, timedelta
from .BndTools import *
from .GridTools import *
from .DischargeTools import GetSrcPoint, ReadSrc, WriteSrc, ReadDis, WriteDis


class D3D():
    """docstring for ."""

    def __init__(self, fname=None):

        self.mdfPath = ''
        self.mdfcheck=True
        if fname:
            _, extention = os.path.splitext(fname.lower())
            if extention != '.mdf':
                raise ValueError("file isn't a mdf file")

            self.filename = fname

            self.mdfPath, _ = os.path.split(fname)

            f = open(fname, 'r')

            for line in f.readlines():

                if '=' in line and 'Commnt' not in line:
                    att = line.split('=')[0].split()[0].capitalize()

                    val = line.split('=')[1].replace('\n', '')

                    if '#' in val:
                        val = val.replace('#', '').rstrip().lstrip()
                    elif len(val.split()) == 1:
                        if '.' in val:
                            val = float(val)
                        else:
                            val = int(val)
                    else:
                        if '.' in val:
                            val = list(map(float, val.split()))
                        else:
                            val = list(map(int, val.split()))

                else:

                    oldval = getattr(self, att)
                    if not isinstance(oldval, list):
                        oldval = [oldval]
                    if len(line.split()) == 1:
                        val = oldval + [float(line.split()[0])]
                setattr(self, att, val)

            # reftime = datetime.strptime(str(self.Itdate), '%Y-%m-%d')
            #
            # self.iniTime = timedelta(minutes=self.Tstart) + reftime
            # self.endTime = timedelta(minutes=self.Tstop) + reftime

            try:
                self.readBnd()
            except BaseException:
                print(os.path.join(self.mdfPath, self.Filbnd) + ' not found')

            try:
                self.readBca()
            except BaseException:
                print('bca file not found')

            try:
                self.readBct()
            except BaseException:
                print('bct file not found')

            try:
                self.readBcc()
            except BaseException:
                print('bcc file not found')
            try:
                self.readGrid()
            except BaseException:
                print('grd file not found')
            try:
                self.readDep()
            except BaseException:
                print('dep file not found')

            try:
                self.readSrc()
            except BaseException:
                print('src file not found')
            try:
                self.readDis()
            except BaseException:
                print('dis file not found')

    def readBnd(self):
        self.bnd = ReadBnd(os.path.join(self.mdfPath, self.Filbnd))

        return self

    def readBca(self):
        self.bca = ReadBca(os.path.join(self.mdfPath, self.Filana))

        return self

    def readBcc(self):

        self.bcc = ReadBctBcc(os.path.join(self.mdfPath, self.Filbcc))

        return self

    def readBct(self):

        self.bct = ReadBctBcc(os.path.join(self.mdfPath, self.Filbct))

        return self

    def readGrid(self):

        self.grd = ReadD3DGrid(os.path.join(self.mdfPath, self.Filcco))

        return self

    def readDep(self):

        self.dep = ReadD3DDep((os.path.join(self.mdfPath, self.Fildep)), ReadD3DGrid(os.path.join(self.mdfPath, self.Filcco)))

        return self

    def readSrc(self):

        self.src = ReadSrc(os.path.join(self.mdfPath, self.Filsrc))

        return self

    def readDis(self):

        self.dis = ReadDis(os.path.join(self.mdfPath, self.Fildis))

        return self

    def write(self, fname=None):

        if not hasattr(self, 'mdfPath'):
            self.mdfPath, _ = os.path.split(fname)

        # reftime = datetime.strptime(str(self.Itdate), '%Y-%m-%d')
        # self.Tstart = self.iniTime - reftime
        # self.Tstart = self.Tstart.seconds / 60 + self.Tstart.days * 24 * 60
        # self.Tstop = self.endTime - reftime
        # self.Tstop = self.Tstop.seconds / 60 + self.Tstop.days * 24 * 60
        if not fname:
            fname = os.path.join(self.mdfPath, self.filename)

        with open(fname, 'w+') as f:
            for key, values in self.__dict__.items():
                if key[0].isupper():
                    if isinstance(values, str):
                        values = "#" + values + "#"
                    elif isinstance(values, list):
                        values = ' '.join(map(str, values))
                    f.write('{0: <6} = {1:}\n'.format(key, values))
        if self.mdfcheck:
            if not os.path.isfile(os.path.join(self.mdfPath, self.Filbnd)):
                WriteBnd(self, os.path.join(self.mdfPath, self.Filbnd))
            if not os.path.isfile(os.path.join(
                    self.mdfPath, self.Filbct)) and self.Filbct:
                WriteBctBcc(self, os.path.join(self.mdfPath, self.Filbct))
            if not os.path.isfile(os.path.join(
                    self.mdfPath, self.Filbcc)) and self.Filbcc:
                WriteBctBcc(self, os.path.join(self.mdfPath, self.Filbcc))
