# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: torchbringer.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x12torchbringer.proto\x12\x19torchbringer.servers.grpc\"\"\n\x06\x43onfig\x12\x18\n\x10serializedConfig\x18\x01 \x01(\t\"\x1c\n\x0c\x43onfirmation\x12\x0c\n\x04info\x18\x01 \x01(\t\",\n\x06Matrix\x12\x12\n\ndimensions\x18\x01 \x03(\x05\x12\x0e\n\x06values\x18\x02 \x03(\x02\"]\n\x07Percept\x12\x30\n\x05state\x18\x01 \x01(\x0b\x32!.torchbringer.servers.grpc.Matrix\x12\x0e\n\x06reward\x18\x02 \x01(\x02\x12\x10\n\x08terminal\x18\x03 \x01(\x08\x32\xc4\x01\n\x15TorchBringerGRPCAgent\x12Z\n\ninitialize\x12!.torchbringer.servers.grpc.Config\x1a\'.torchbringer.servers.grpc.Confirmation\"\x00\x12O\n\x04step\x12\".torchbringer.servers.grpc.Percept\x1a!.torchbringer.servers.grpc.Matrix\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'torchbringer_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_CONFIG']._serialized_start=49
  _globals['_CONFIG']._serialized_end=83
  _globals['_CONFIRMATION']._serialized_start=85
  _globals['_CONFIRMATION']._serialized_end=113
  _globals['_MATRIX']._serialized_start=115
  _globals['_MATRIX']._serialized_end=159
  _globals['_PERCEPT']._serialized_start=161
  _globals['_PERCEPT']._serialized_end=254
  _globals['_TORCHBRINGERGRPCAGENT']._serialized_start=257
  _globals['_TORCHBRINGERGRPCAGENT']._serialized_end=453
# @@protoc_insertion_point(module_scope)