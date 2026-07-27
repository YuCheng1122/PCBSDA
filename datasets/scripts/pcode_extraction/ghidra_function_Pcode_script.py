import os
import json
import logging
from ghidra.app.util.headless import HeadlessScript
from ghidra.program.model.symbol import Reference
from ghidra.app.decompiler import DecompInterface

# Get script arguments and determine the save folder
currentProgram = getCurrentProgram()
argv = getScriptArgs()

try:
    # Set save folder
    if len(argv) == 2:
        output_folder = argv[0]
        results_folder = argv[1]
    elif len(argv) == 1:
        output_folder = argv[0]
        results_folder = os.path.join(output_folder, 'results')
    elif len(argv) == 0:
        output_folder = os.getcwd()
        results_folder = os.path.join(os.getcwd(), 'results')
    else:
        raise ValueError("Invalid number of arguments")
except Exception as e:
    error_message = "An error occurred while setting parameters: {}".format(e)
    logging.error(error_message, exc_info=True)

program_name = currentProgram.getName()
program_folder = os.path.join(results_folder, program_name)

# Create the program-specific directory
if not os.path.exists(program_folder):
    os.makedirs(program_folder)

# Set up logging
log_file_path = os.path.join(output_folder, 'extraction.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Determine file paths for DOT file and JSON file
dot_file_path = os.path.join(program_folder, program_name + '.dot')
json_file_path = os.path.join(program_folder, program_name + '.json')

try:
    fm = currentProgram.getFunctionManager()
    funcs = fm.getFunctions(True)

    dot_lines = ["digraph code {"]
    functions_info = {}
    
    # Initialize the decompiler for extracting Hign Level Pcode
    decompiler = DecompInterface()
    decompiler.openProgram(currentProgram)

    # Collecting all lines to write in batch
    for func in funcs:
        if func.isExternal() or func.isThunk():
            continue
        entry_point = func.getEntryPoint()
        entry_point_offset = hex(entry_point.getOffset())
        name = func.getName()
        
        decompile_results = decompiler.decompileFunction(func, 30, None)
        hign_func = decompile_results.getHighFunction()
        if hign_func is None:
            logging.warning("Decompilation failed for function: {}".format(name))
            continue

        # Prepare function information for JSON
        functions_info[entry_point_offset] = {
            "function_name": name,
            "instructions": []
        }

        dot_lines.append('  "{}" [label="{}"];'.format(entry_point_offset, name))

        # Extracting instructions for each function
        for pcode_op in hign_func.getPcodeOps():
            instruction_info = {
                "address": str(pcode_op.getSeqnum().getTarget()),
                "operation": str(pcode_op),
                "opcode": pcode_op.getMnemonic()
            }
            functions_info[entry_point_offset]["instructions"].append(instruction_info)

        callees = func.getCalledFunctions(None)
        for callee in callees:
            # get the entry point of the callee function
            if callee.isExternal() or callee.isThunk():
                continue
            callee_entry_point = callee.getEntryPoint()
            callee_entry_point_offset = hex(callee_entry_point.getOffset())
            # Write DOT file content
            dot_lines.append('  "{}" -> "{}";'.format(entry_point_offset, callee_entry_point_offset))

    dot_lines.append("}")

    # Writing to DOT file
    with open(dot_file_path, "w") as dot_file:
        dot_file.write("\n".join(dot_lines))

    # Writing to JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(functions_info, json_file, indent=4)

except Exception as e:
    error_message = "An error occurred while writing the files: {}".format(e)
    logging.error(error_message, exc_info=True)
    decompiler.closeProgram()
