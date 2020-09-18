import os
import numpy as np
from hls4ml.writer.vivado_writer import VivadoWriter
from hls4ml.model.hls_model import IntegerPrecisionType, FixedPrecisionType

class PynqWriter(VivadoWriter):

    def next_axi_type(self, p):
        ''' Return a new type with the width rounded to the next factor of 8 up to p's width
            Args:
                p : IntegerPrecisionType or FixedPrecisionType
            Returns:
                An IntegerPrecisionType or FixedPrecisionType with the width rounder up to the next factor of 8
                of p's width. Other parameters (fractional bits, extra modes) stay the same.
        '''
        W = p.width
        newW = int(np.ceil(W / 8) * 8)
        if isinstance(p, FixedPrecisionType):
            return FixedPrecisionType(newW, p.integer, p.signed, p.rounding_mode, p.saturation_mode, p.saturation_bits)
        elif isinstance(p, IntegerPrecisionType):
            return IntegerPrecisionType(newW, p.signed)


    def write_axi_wrapper(self, model):
        ''' Write a top level HLS C++ file to wrap the hls4ml project with AXI interfaces
            Args:
                model : The HLSModel to write the wrapper for
        '''

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        assert len(model_inputs) == 1, "Only models with one input tensor are currently supported by PynqBackend"
        assert len(model_outputs) == 1, "Only models with one output tensor are currently supported by PynqBackend"
        inp = model_inputs[0]
        out = model_outputs[0]
        inp_axi_t = self.next_axi_type(inp.type.precision)
        out_axi_t = self.next_axi_type(inp.type.precision)

        indent = '    '

        #######################
        ## myproject_axi.h
        #######################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/pynq/myproject_axi.h'),'r')
        fout = open('{}/firmware/{}_axi.h'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT',format(model.config.get_project_name().upper()))
            elif 'void myproject(' in line:
                newline = 'void {}_axi(\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert definitions' in line:
                newline = ''
                newline += 'static const unsigned N_IN = {};\n'.format(inp.size())
                newline += 'static const unsigned N_OUT = {};\n'.format(out.size())
                newline += 'typedef {} input_axi_t;\n'.format(inp_axi_t)
                newline += 'typedef {} output_axi_t;\n'.format(out_axi_t)
                #newline += 'typedef {} input_t;\n'.format(inp.type.precision)
                #newline += 'typedef {} output_t ;\n'.format(out.type.precision)
                #newline += 'typedef {} input_axi_t;\n'.format(inp_axi_t)
                #newline += 'typedef {} output_axi_t;\n'.format(out_axi_t)
                #newline += 'typedef {} input_t;\n'.format(inp.type.precision)
                #newline += 'typedef {} output_t;\n'.format(out.type.precision)
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

        #######################
        ## myproject_axi.cpp
        #######################

        f = open(os.path.join(filedir,'../templates/pynq/myproject_axi.cpp'),'r')
        fout = open('{}/firmware/{}_axi.cpp'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        for line in f.readlines():
            if 'void myproject(' in line:
                newline = 'void {}_axi(\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert local vars' in line:
                newline = ''
                newline += indent + inp.type.name + ' in_local[N_IN];\n'
                newline += indent + out.type.name + ' out_local[N_OUT];\n'
            elif '//hls-fpga-machine-learning insert call' in line:
                newline = indent + '{}(in_local, out_local, in_size, out_size);\n'.format(model.config.get_project_name())
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()

    def modify_build_script(self, model):
        '''
        Modify the build_prj.tcl script to add the extra wrapper files and set the top function
        '''
        filedir = os.path.dirname(os.path.abspath(__file__))
        oldfile = '{}/build_prj.tcl'.format(model.config.get_output_dir())
        newfile = '{}/build_prj_axi.tcl'.format(model.config.get_output_dir())
        f = open(oldfile,'r')
        fout = open(newfile, 'w')

        for line in f.readlines():
            if 'set_top' in line:
                newline = line[:-1] + '_axi\n' # remove the newline from the line end and append _axi for the new top
                #newline += 'add_files firmware/{}_axi.h -cflags "-std=c++0x"\n'.format(model.config.get_project_name())
                newline += 'add_files firmware/{}_axi.cpp -cflags "-std=c++0x"\n'.format(model.config.get_project_name())
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()
        os.rename(newfile, oldfile)
        
    def write_hls(self, model):
        '''
        Write the HLS project. Calls the VivadoBackend writer, and extra steps for Pynq/AXI interface
        '''
        super(PynqWriter, self).write_hls(model)
        self.write_axi_wrapper(model)
        self.modify_build_script(model)


