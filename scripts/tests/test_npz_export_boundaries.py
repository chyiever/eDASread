"""Boundary regressions for eDAS NPZ and source-preserving stride exports.

Run from project root: python -m unittest scripts.tests.test_npz_export_boundaries -v
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from scripts.core.models import FileMetadata
from scripts.io.bin_reader import inspect_bin_file
from scripts.io.npz_codec import read_edas_npz, write_edas_npz
from scripts.io.view_export import ViewExportRequest, export_view


class NpzExportBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="edas_npz_boundary_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.raw = np.array([[np.iinfo(np.int32).min, 0, np.iinfo(np.int32).max],
                             [2**25+1, -2**25-1, 17], [31,41,59], [26,53,58], [97,93,23]], dtype=np.int32)
        self.source = self.root / "eDAS-1000Hz-0003pt-20261016T120000.000.bin"
        self.raw.tofile(self.source)
        self.meta = inspect_bin_file(self.source)
        self.request = ViewExportRequest(self.meta, (self.meta,), None, 0, 5, 0, 3, 1, 1, "npz", False, ())
        self.header = {
            "data_type":"eDAS", "format_version":1, "dtype":"int32", "raw_unfiltered":True,
            "frame_count":5, "channel_count":3, "sample_rate_hz":1000,
            "start_time":"2026-10-16T12:00:00", "acquisition_start_time":"2026-10-16T12:00:00",
            "channel_start":0, "channel_step":1, "channel_numbers":[0,1,2], "arrivals":[],
            "arrival_time_reference":"seconds_from_first_sample",
        }

    def load_export(self, request):
        result = export_view(request, self.output)
        data, header = read_edas_npz(result["output_path"])
        return data, header, result

    def test_full_int32_range_is_lossless(self):
        data, _, result = self.load_export(self.request)
        np.testing.assert_array_equal(data,self.raw)
        self.assertEqual(result["dtype"],"int32")

    def test_one_sample_one_channel_and_large_steps(self):
        r=replace(self.request,frame_start=4,column_start=2,time_step=100,space_step=100,arrivals=((2,.004),))
        data,h,_=self.load_export(r)
        np.testing.assert_array_equal(data,self.raw[4:5,2:3])
        self.assertEqual(h["arrivals"],[{"channel":2,"time_s":0.0}])
        self.assertEqual(h["channel_count"],1)
        self.assertEqual(h["sample_rate_hz"],10)

    def test_unknown_acquisition_time_is_null(self):
        meta=replace(self.meta,start_time=None,acquisition_start_time=None)
        _,h,_=self.load_export(replace(self.request,metadata=meta))
        self.assertIsNone(h["start_time"])
        self.assertIsNone(h["acquisition_start_time"])

    def test_arrivals_include_bounds_and_exclude_removed_samples(self):
        r=replace(self.request,frame_start=1,time_step=2,space_step=2,
                  arrivals=((0,.001),(2,.003),(1,.002)))
        data,h,_=self.load_export(r)
        np.testing.assert_array_equal(data,self.raw[1:5:2,::2])
        self.assertEqual(h["arrivals"],[{"channel":0,"time_s":0.0},{"channel":2,"time_s":.002}])
        # The last raw sample is removed by time stride, so its pick is excluded.
        _,h,_=self.load_export(replace(r,arrivals=((0,.004),)))
        self.assertEqual(h["arrivals"],[])

    def test_stride_phase_across_unequal_source_lengths(self):
        matrices=[np.arange(n*3,dtype=np.int32).reshape(n,3)+100*i for i,n in enumerate((2,5,4))]
        sources=[]
        for i,data in enumerate(matrices):
            path=self.root/f"source{i}.bin";data.tofile(path)
            sources.append(replace(self.meta,file_path=path,frame_count=len(data),duration_s=len(data)/1000,file_size_bytes=data.nbytes))
        all_data=np.concatenate(matrices)
        meta=replace(self.meta,frame_count=11,duration_s=.011)
        r=replace(self.request,metadata=meta,sources=tuple(sources),frame_start=1,frame_end=11,time_step=3)
        actual,_,_=self.load_export(r)
        np.testing.assert_array_equal(actual,all_data[1:11:3])

    def test_npz_ignores_processed_flag_and_matrix(self):
        data,_,result=self.load_export(replace(self.request,processed=True,data=np.full(self.raw.shape,999,dtype=np.float32)))
        np.testing.assert_array_equal(data,self.raw)
        self.assertFalse(result["filtered"])

    def test_processed_txt_uses_processed_data(self):
        phase=np.arange(15,dtype=np.float32).reshape(5,3)/10
        r=replace(self.request,format="txt",processed=True,data=phase,time_step=2,space_step=2)
        result=export_view(r,self.output)
        np.testing.assert_allclose(np.loadtxt(result["output_path"]),phase[::2,::2])
        self.assertTrue(result["filtered"])

    def test_invalid_steps_and_empty_ranges_are_rejected(self):
        for field in ("time_step","space_step"):
            for value in (0,-1,1.5,True):
                with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                    export_view(replace(self.request,**{field:value}),self.output)
        for changes in ({"frame_end":0},{"column_end":0},{"frame_end":6}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                export_view(replace(self.request,**changes),self.output)

    def test_source_cannot_be_overwritten(self):
        original=self.source.read_bytes()
        with self.assertRaises(ValueError):
            export_view(replace(self.request,format="bin"),self.root)
        self.assertEqual(self.source.read_bytes(),original)

    def test_invalid_headers_and_objects_are_rejected(self):
        changes=(
            {"data_type":"other"}, {"format_version":True}, {"frame_count":5.0},
            {"channel_start":False}, {"channel_step":0}, {"sample_rate_hz":float('nan')},
            {"channel_numbers":[False,1,2]}, {"start_time":"invalid"},
            {"arrivals":[{"channel":0,"time_s":-.1}]},
            {"arrivals":[{"channel":0,"time_s":float('inf')}]},
            {"arrivals":[{"channel":0,"time_s":0},{"channel":0,"time_s":.001}]},
        )
        for i,change in enumerate(changes):
            with self.subTest(change=change):
                header=dict(self.header,**change)
                path=self.root/f"bad{i}.npz"
                np.savez(path,data=self.raw,header=np.array(json.dumps(header)))
                with self.assertRaises(ValueError):read_edas_npz(path)
        path=self.root/"object.npz"
        np.savez(path,data=np.array([[object()]],dtype=object),header=np.array(json.dumps(self.header)))
        with self.assertRaises(ValueError):read_edas_npz(path)

    def test_reexport_retains_acquisition_origin_and_channel_mapping(self):
        h=copy.deepcopy(self.header)
        h.update(channel_start=20,channel_step=3,channel_numbers=[20,23,26],arrivals=[{"channel":23,"time_s":.002}])
        path=self.root/'input.npz';write_edas_npz(path,self.raw,h)
        metadata=inspect_bin_file(path)
        r=replace(self.request,metadata=metadata,sources=(metadata,),frame_start=1,column_start=1,
                  arrivals=metadata.arrivals,time_step=2)
        data,head,_=self.load_export(r)
        np.testing.assert_array_equal(data,self.raw[1::2,1:])
        self.assertEqual(head["channel_numbers"],[23,26])
        self.assertEqual(head["start_time"],"2026-10-16T12:00:00.001000")
        self.assertEqual(head["acquisition_start_time"],h["acquisition_start_time"])
        self.assertEqual(head["arrivals"],[{"channel":23,"time_s":.001}])


if __name__=="__main__":
    unittest.main(verbosity=2)
