from obspy import read, Trace, Stream, UTCDateTime
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader
from obspy.io.segy.core import _read_segy
import numpy as np
import sys
# from six.moves import xrange

st0 = read()
stream = Stream()
print(st0)
st0.detrend('spline',order = 3, dspline=100)
# st0.plot()
data = st0[0].data
print(len(data))
path = "D:/OneDrive - EVER/半导体所/数据处理/DAS-代码/SemiTDM-DAS-100KHz 2023-8-8-1-24-14.732-9trace.txt"
data9 = np.loadtxt(path)

for i in range(3):
    # Create some random data.
    # data = st0[i].data
    # data = data9[:100000,i]
    # print(len(data))

    # data = np.unwrap( data * np.pi )
    # data = np.require(data, dtype=np.float32)
    
    # trace = Trace(data = data)
    data = np.require(st0[i].data, dtype=np.float32)

    trace = Trace(data = data)
    
    # trace.decimate(factor=100, no_filter=True, strict_length=False)
    # trace.detrend('spline',order = 3, dspline = 15)
    trace.stats.delta = 0.01#00#0.01
    trace.stats.starttime = UTCDateTime(2011,11,11,11,11,11)
    if not hasattr(trace.stats, 'segy.trace_header'):
        trace.stats.segy = {}
    trace.stats.segy.trace_header = SEGYTraceHeader()
    trace.stats.segy.trace_header.trace_sequence_number_within_line = i + 1
    trace.stats.segy.trace_header.receiver_group_elevation = 444

    # Add trace to stream
    stream.append(trace)

stream.stats = AttribDict()
stream.stats.textual_file_header = 'Textual Header!'
stream.stats.binary_file_header = SEGYBinaryFileHeader()
stream.stats.binary_file_header.trace_sorting_code = 5
print ("Stream object before writing...")
print(stream)
stream.write("TEST9.sgy", format="SEGY", data_encoding=1,             byteorder=sys.byteorder)
print ("Stream object after writing. Will have some segy attributes...")


###================================================================================










# stream = Stream()

# print(stream)
# for i in range(3):
#     # Create some random data.
#     data = np.random.ranf(1000)
#     print(i,data.size)
#     data = np.require(data, dtype=np.float32)
#     trace = Trace(data=data)
    
#     # Attributes in trace.stats will overwrite everything in
#     # trace.stats.segy.trace_header
#     trace.stats.delta = 0.01
#     # SEGY does not support microsecond precision! Any microseconds will
#     # be discarded.
#     trace.stats.starttime = UTCDateTime(2011,11,11,11,11,11)

#     # If you want to set some additional attributes in the trace header,
#     # add one and only set the attributes you want to be set. Otherwise the
#     # header will be created for you with default values.
#     if not hasattr(trace.stats, 'segy.trace_header'):
#         trace.stats.segy = {}
#     trace.stats.segy.trace_header = SEGYTraceHeader()
#     trace.stats.segy.trace_header.trace_sequence_number_within_line = i + 1
#     trace.stats.segy.trace_header.receiver_group_elevation = 444

#     # Add trace to stream
#     stream.append(trace)

# # A SEGY file has file wide headers. This can be attached to the stream
# # object.  If these are not set, they will be autocreated with default
# # values.
# stream.stats = AttribDict()
# stream.stats.textual_file_header = 'Textual Header!'
# stream.stats.binary_file_header = SEGYBinaryFileHeader()
# stream.stats.binary_file_header.trace_sorting_code = 5

# print ("Stream object before writing...")
# print (stream)

# stream.write("TEST.sgy", format="SEGY", data_encoding=1,
#              byteorder=sys.byteorder)
# print ("Stream object after writing. Will have some segy attributes...")
# print (stream)

# print ("Reading using obspy.io.segy...")
# st1 = _read_segy("TEST.sgy")
# print (st1)

# print( "Reading using obspy.core...")
# st2 = read("TEST.sgy")
# print (st2)

# print("Just to show that the values are written...")
# print( [tr.stats.segy.trace_header.receiver_group_elevation
#        for tr in stream])
# print([tr.stats.segy.trace_header.receiver_group_elevation for tr in st2])
# print(stream.stats.binary_file_header.trace_sorting_code)
# print(st1.stats.binary_file_header.trace_sorting_code)



