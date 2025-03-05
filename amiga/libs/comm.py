#!/usr/bin/env python

'''
AMiGA library of functions for printing and communicating via terminal.
'''

__author__ = "Firas S Midani"
__email__ = "midani@bcm.edu"


# TABLE OF CONTENTS (4 functions)

# smartPrint
# prettyNumberDisplay
# tidyDictPrint
# tidyMessage

import numpy as np # type: ignore


def smartPrint(msg,verbose):
    '''
    Only print if verbose argument is True. 

        Previously, I would concatenate messages inside a function and print once 
        function is completed, if verbose argument is satisfied. But the side 
        effect is that messages are printed in blocks (i.e. not flushing)
        and printing would not execute.

        In incremental printing (which this streamlines), if a function fails at a 
        specific point, this can be inferred based on where incremental printing would
        have been interrupted right after the point of failure.

    Args:
        msg (str)
        verbose (boolean)
    '''

    if verbose:
        print(msg)

def prettyNumberDisplay(num):
    '''
    Displays numebrs that are either large or small in scientific noation, otherwise does not.

    Args:
        num (float or int)
    '''

    if np.floor(np.abs(np.log10(np.abs(num)))) > 4:
        return f'{num:.3}'
    else:
        return f'{num:.3f}'


def tidyDictPrint(input_dict):
    '''
    Returns a message that neatly prints a dictionary into multiple lines. Each line
        is a key:value pair. Keys of dictionary are padded with period. Padding is 
        dynamically selected based on longest argument.
    '''

    if len(input_dict)==0:
        return 'None' 

    # dynamically set width of padding based on maximum argument length
    args = input_dict.keys()
    max_len = float(len(max(args,key=len))) # length of longest argument
    width = int(np.ceil(max_len/10)*10) # round up length to nearest ten

    # if width dos not add more than three padding spaces to argument, add 5 characters
    if (width - max_len) < 4:
        width += 5

    # compose multi-line message
    msg = ''
    for arg,value in input_dict.items():
        msg += '{:.<{width}}{}\n'.format(arg,value,width=width)

    return msg


def tidyMessage(msg):
    '''
    Returns a message in a text-based banner with header and footer
        composed of dashes and hashtags. Messae will be paddedd with flanking
        spaces. Length of message determines length of banner. Heigh of banner is 4 lines.

    Args:
        msg (str)

    Returns:
        msg_print (str)
    '''

    msg_len = len(msg)+2
    banner = '#{}#'.format('-'*msg_len)
    msg_print = f'{banner}\n# {msg} #\n{banner}\n'

    return msg_print

