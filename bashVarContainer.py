#This python object is designed to interact with a file that holds variables to be accessed from a bash shell.
#A bash variable file is a plaintext file with a column of keys and a column of vals, separated by colons
#Once such a file exists, (call it bashVarFile.txt in directory /Users/sasha/dir/with/bash/var/), a user
# can pull the value of variable wantedVar from the file via the following command:
#_________________________________________________
# $ storedVarVal=$(python bashVarContainer.py bashVarFile.txt wantedVar /Users/sasha/dir/with/bash/var/)
#_________________________________________________
#The directory can be left blanck.

import sys
import cantrips as c

def getVarFromDictBash(container_file, var_key, container_dir = ''):
    bashContainer = bashVarContainerObject(container_file, container_dir = container_dir, readInFromFile = True)
    if bashContainer.containsVar(var_key):
        bashContainer.getVarFromDictBash(var_key)
        return 0
    else:
        print ('The variable container file ' + str(container_dir + container_file) + ' does not have the requested key ' + str(var_key))
        return 1

class bashVarContainerObject:

    def containsVar(self, var_key):
        if var_key in self.var_dict.keys():
            return 1
        else:
            return 0

    def updateVarDict(self, var_key, var_val):
        self.var_dict[var_key] = var_val
        self.dict_len = self.dict_len + 1
        return 0

    def getVarFromDictPython(self, var_key):
        val = self.var_dict[var_key]
        return val

    def getVarFromDictBash(self, var_key):
        val = self.var_dict[var_key]
        print(val)
        return 0

    def readInFromFile (self):
        #print (c.readInColumnsToList(self.container_file, file_dir = self.container_dir, delimiter = ':', n_ignore = 1, verbose = 0))
        keys, vals = c.readInColumnsToList(self.container_file, file_dir = self.container_dir, delimiter = self.delimiter, n_ignore = 1, verbose = 0)
        #print ('key_val_pairs = ' + str(key_val_pairs))
        #keys = [key_val_pair[0] for key_val_pair in key_val_pairs]
        #print ('keys = ' +str(keys))
        #for key_val_pair in key_val_pairs:
        #    print ('key_val_pair = ' + str(key_val_pair))
        #    print ('key_val_pair[1] = ' + str(key_val_pair[1]))
        #vals = [key_val_pair[1] for key_val_pair in key_val_pairs]
        n_keys = len(keys)
        if not (self.key_types is None):
            keys = [key_types[i](keys[i]) for i in range(n_keys)]
        if not (self.val_types is None):
            vals = [val_types[i](vals[i]) for i in range(n_keys)]
        self.var_dict = {keys[i]:vals[i] for i in range(n_keys)}
        return 0

    def saveContainerToFile(self, note_type = True):
        keys = list(self.var_dict.keys())
        vals = [self.var_dict[key] for key in keys]
        if note_type:
            self.key_types = [type(key) for key in keys]
            self.val_types = [type(self.var_dict[key]) for key in keys]
        header = ['keys', 'vals']
        return c.saveListsToColumns([keys, vals], self.container_file, self.container_dir, sep = ':', append = False, header = header)

    def __init__(self, container_file, var_dict = {}, container_dir = '', readInFromFile = False, delimiter = ':'):

        self.container_file = container_file
        self.container_dir = container_dir
        self.var_dict = var_dict
        self.key_types = None
        self.val_types = None
        self.delimiter = delimiter

        if readInFromFile:
            self.readInFromFile()
        else:
            self.var_dict = var_dict
        self.dict_len = len(list(self.var_dict.keys()))



if __name__=="__main__":
    command_line_args = sys.argv[1:]
    if len(command_line_args) > 2:
        container_file, var_key, container_dir = command_line_args
    else:
        container_file, var_key = command_line_args
        container_dir = ''
    getVarFromDictBash(container_file, var_key, container_dir = container_dir)
