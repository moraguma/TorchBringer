# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: torchdeeprl.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11torchdeeprl.proto\x12\x0btorchdeeprl\"\"\n\x06\x43onfig\x12\x18\n\x10serializedConfig\x18\x01 \x01(\t\"\x1c\n\x0c\x43onfirmation\x12\x0c\n\x04info\x18\x01 \x01(\t\",\n\x06Matrix\x12\x12\n\ndimensions\x18\x01 \x03(\x05\x12\x0e\n\x06values\x18\x02 \x03(\x02\"O\n\x07Percept\x12\"\n\x05state\x18\x01 \x01(\x0b\x32\x13.torchdeeprl.Matrix\x12\x0e\n\x06reward\x18\x02 \x01(\x02\x12\x10\n\x08terminal\x18\x03 \x01(\x08\x32\x86\x01\n\x0fTorchDeepRLGrpc\x12>\n\ninitialize\x12\x13.torchdeeprl.Config\x1a\x19.torchdeeprl.Confirmation\"\x00\x12\x33\n\x04step\x12\x14.torchdeeprl.Percept\x1a\x13.torchdeeprl.Matrix\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'torchdeeprl_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_CONFIG']._serialized_start=34
  _globals['_CONFIG']._serialized_end=68
  _globals['_CONFIRMATION']._serialized_start=70
  _globals['_CONFIRMATION']._serialized_end=98
  _globals['_MATRIX']._serialized_start=100
  _globals['_MATRIX']._serialized_end=144
  _globals['_PERCEPT']._serialized_start=146
  _globals['_PERCEPT']._serialized_end=225
  _globals['_TORCHDEEPRLGRPC']._serialized_start=228
  _globals['_TORCHDEEPRLGRPC']._serialized_end=362
# @@protoc_insertion_point(module_scope)