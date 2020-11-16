import serial
import json

from lib.shell import load_config, send_config, show_config
import mmw_1443 as mss
import plotter

control_port = "COM4"
data_port = "COM3"

try:
    control_serial = serial.Serial(control_port, 115200, timeout=0.01)
    data_serial = serial.Serial(data_port, 921600, timeout=0.01)
except serial.serialutil.SerialException:
    print("Cannot connect to the ports")

if control_serial is not None and data_serial is not None:
    config_file = open("mss/14_mmw-xWR14xx.cfg", "r")
    content = load_config(config_file)
    cfg = json.loads(content)
    cfg, par = mss.config(cfg)
    send_config(control_serial, cfg)
    show_config(cfg)

    input_buffer, output, sync, size = {'buffer': b''}, {}, False, mss.meta['blk']
    logger = open("data.json", "a")

    while True:
        try:
            data = data_serial.read(size)
            input_buffer['buffer'] += data

            if data[:len(mss.meta['seq'])] == mss.meta['seq']:  # check for magic sequence
                if len(output) > 0:
                    plain = json.dumps(output)
                    plotter.update_plot(plain)
                    logger.write(plain + "\n")

                input_buffer['buffer'] = data
                input_buffer['blocks'] = -1
                input_buffer['address'] = 0
                input_buffer['values'] = 0
                input_buffer['other'] = {}

                output = {}
                sync = True  # very first frame in the stream was seen

            if sync:
                buffer_length = 0
                while buffer_length < len(input_buffer['buffer']):  # keep things finite
                    buffer_length = len(input_buffer['buffer'])
                    mss.aux_buffer(input_buffer, output)  # do processing of captured bytes
        except KeyboardInterrupt:
            control_serial.close()
