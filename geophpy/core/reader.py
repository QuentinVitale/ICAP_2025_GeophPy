# -*- coding: utf-8 -*-.
'''
    geophpy.core.reader
    ---------------------------

    Management of basic input and output ascii file format.

    :copyright: Copyright 2019 Quentin Vitale, Lionel Darras, Philippe Marty and contributors, see AUTHORS.
    :license: GNU GPL v3.
    :author: Quentin VITALE
    :license: GNU GPL v3.
    :version: 0.1
    :revision: 2019/05/05

'''

import csv

class Data:
    ''' Base class to store data '''
    pass

class AsciiDataFile:
    ''' Base class to define ASCII data file. '''

    def __init__(self, filename):

        if not filename.endswith(self.ext):
            raise Exception("Invalid file format")

        self.filename = filename

class RowReader:
    ''' Base class to read delimiter-separated values files into a list. '''

    def __init__(self, filename=None, delimiter='\t'):
        self.filename = filename
        self.delimiter = delimiter
        self.Nrow = None
        self.rows = []

    def sniff(self):
        ''' Sniff the delimiter from the dsv file. '''

        if self.filename is not None:
            with open(self.filename, 'r', newline="") as csvfile:
                dialect = csv.Sniffer().sniff(csvfile.readline())
                if dialect.delimiter:
                    self.delimiter = dialect.delimiter

        return self

    def read(self):
        ''' Read dsv file line-by-line and store it in a list. '''

        # Sniffing file delimiter
        if self.delimiter is None:
            self.sniff()

        # Reading dsv file's rows into a list
        if self.delimiter is not None:
            with open(self.filename, 'r', newline="") as csvfile:
                csvreader = csv.reader(csvfile, delimiter=self.delimiter)

                for row in csvreader:
                    self.rows.append(row)

            self.Nrows = len(self.rows)

        return self

class RowWriter:
    ''' Base class to write list into a delimiter-separated values file. '''

    def __init__(self, filename=None, rows=[], delimiter='\t'):
        self.filename = filename
        self.delimiter = delimiter
        self.rows = rows
        self.Nrow = len(rows)

    def write(self, filename=None, delimiter=None):
        ''' Write rows into a ds file. '''
        # Default name
        if delimiter is not None:
            self.delimiter = delimiter
        if filename is not None:
            self.filename = filename
        
        if self.filename is not None:
                filename = self.filename
        else:
                self.filename = 'out_georef.txt'
                filename = self.filename

        # Writting file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.delimiter)

            for row in self.rows:
                writer.writerow(row)

        return self

