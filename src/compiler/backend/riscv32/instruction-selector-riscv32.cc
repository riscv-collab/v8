// Copyright 2021 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/base/bits.h"
#include "src/compiler/backend/instruction-selector-impl.h"
#include "src/compiler/node-matchers.h"
#include "src/compiler/node-properties.h"

namespace v8 {
namespace internal {
namespace compiler {

#define TRACE_UNIMPL() \
  PrintF("UNIMPLEMENTED instr_sel: %s at line %d\n", __FUNCTION__, __LINE__)

#define TRACE() PrintF("instr_sel: %s at line %d\n", __FUNCTION__, __LINE__)

// Adds RISC-V-specific methods for generating InstructionOperands.
class RiscvOperandGenerator final : public OperandGenerator {
 public:
  explicit RiscvOperandGenerator(InstructionSelector* selector)
      : OperandGenerator(selector) {}

  InstructionOperand UseOperand(Node* node, InstructionCode opcode) {
    if (CanBeImmediate(node, opcode)) {
      return UseImmediate(node);
    }
    return UseRegister(node);
  }

  // Use the zero register if the node has the immediate value zero, otherwise
  // assign a register.
  InstructionOperand UseRegisterOrImmediateZero(Node* node) {
    if ((IsIntegerConstant(node) && (GetIntegerConstantValue(node) == 0)) ||
        (IsFloatConstant(node) &&
         (bit_cast<int64_t>(GetFloatConstantValue(node)) == 0))) {
      return UseImmediate(node);
    }
    return UseRegister(node);
  }

  bool IsIntegerConstant(Node* node) {
    return (node->opcode() == IrOpcode::kInt32Constant);
  }

  int64_t GetIntegerConstantValue(Node* node) {
    DCHECK_EQ(IrOpcode::kInt32Constant, node->opcode());
    return OpParameter<int32_t>(node->op());
  }

  bool IsFloatConstant(Node* node) {
    return (node->opcode() == IrOpcode::kFloat32Constant) ||
           (node->opcode() == IrOpcode::kFloat64Constant);
  }

  double GetFloatConstantValue(Node* node) {
    if (node->opcode() == IrOpcode::kFloat32Constant) {
      return OpParameter<float>(node->op());
    }
    DCHECK_EQ(IrOpcode::kFloat64Constant, node->opcode());
    return OpParameter<double>(node->op());
  }

  bool CanBeImmediate(Node* node, InstructionCode mode) {
    return IsIntegerConstant(node) &&
           CanBeImmediate(GetIntegerConstantValue(node), mode);
  }

  bool CanBeImmediate(int64_t value, InstructionCode opcode) {
    switch (ArchOpcodeField::decode(opcode)) {
      case kRiscvShl32:
      case kRiscvSar32:
      case kRiscvShr32:
        return is_uint5(value);
      case kRiscvAdd:
      case kRiscvAnd32:
      case kRiscvAnd:
      case kRiscvOr32:
      case kRiscvOr:
      case kRiscvTst:
      case kRiscvXor:
        return is_int12(value);
      case kRiscvLb:
      case kRiscvLbu:
      case kRiscvSb:
      case kRiscvLh:
      case kRiscvLhu:
      case kRiscvSh:
      case kRiscvLw:
      case kRiscvSw:
      case kRiscvLoadFloat:
      case kRiscvStoreFloat:
      case kRiscvLoadDouble:
      case kRiscvStoreDouble:
        return is_int32(value);
      default:
        return is_int12(value);
    }
  }

 private:
  bool ImmediateFitsAddrMode1Instruction(int32_t imm) const {
    TRACE_UNIMPL();
    return false;
  }
};

static void VisitRR(InstructionSelector* selector, ArchOpcode opcode,
                    Node* node) {
  RiscvOperandGenerator g(selector);
  selector->Emit(opcode, g.DefineAsRegister(node),
                 g.UseRegister(node->InputAt(0)));
}

static void VisitRRI(InstructionSelector* selector, ArchOpcode opcode,
                     Node* node) {
  RiscvOperandGenerator g(selector);
  int32_t imm = OpParameter<int32_t>(node->op());
  selector->Emit(opcode, g.DefineAsRegister(node),
                 g.UseRegister(node->InputAt(0)), g.UseImmediate(imm));
}

static void VisitSimdShift(InstructionSelector* selector, ArchOpcode opcode,
                           Node* node) {
  RiscvOperandGenerator g(selector);
  if (g.IsIntegerConstant(node->InputAt(1))) {
    selector->Emit(opcode, g.DefineAsRegister(node),
                   g.UseRegister(node->InputAt(0)),
                   g.UseImmediate(node->InputAt(1)));
  } else {
    selector->Emit(opcode, g.DefineAsRegister(node),
                   g.UseRegister(node->InputAt(0)),
                   g.UseRegister(node->InputAt(1)));
  }
}

static void VisitRRIR(InstructionSelector* selector, ArchOpcode opcode,
                      Node* node) {
  RiscvOperandGenerator g(selector);
  int32_t imm = OpParameter<int32_t>(node->op());
  selector->Emit(opcode, g.DefineAsRegister(node),
                 g.UseRegister(node->InputAt(0)), g.UseImmediate(imm),
                 g.UseRegister(node->InputAt(1)));
}

static void VisitRRR(InstructionSelector* selector, ArchOpcode opcode,
                     Node* node) {
  RiscvOperandGenerator g(selector);
  selector->Emit(opcode, g.DefineAsRegister(node),
                 g.UseRegister(node->InputAt(0)),
                 g.UseRegister(node->InputAt(1)));
}

static void VisitUniqueRRR(InstructionSelector* selector, ArchOpcode opcode,
                           Node* node) {
  RiscvOperandGenerator g(selector);
  selector->Emit(opcode, g.DefineAsRegister(node),
                 g.UseUniqueRegister(node->InputAt(0)),
                 g.UseUniqueRegister(node->InputAt(1)));
}

void VisitRRRR(InstructionSelector* selector, ArchOpcode opcode, Node* node) {
  RiscvOperandGenerator g(selector);
  selector->Emit(
      opcode, g.DefineSameAsFirst(node), g.UseRegister(node->InputAt(0)),
      g.UseRegister(node->InputAt(1)), g.UseRegister(node->InputAt(2)));
}

static void VisitRRO(InstructionSelector* selector, ArchOpcode opcode,
                     Node* node) {
  RiscvOperandGenerator g(selector);
  selector->Emit(opcode, g.DefineAsRegister(node),
                 g.UseRegister(node->InputAt(0)),
                 g.UseOperand(node->InputAt(1), opcode));
}

bool TryMatchImmediate(InstructionSelector* selector,
                       InstructionCode* opcode_return, Node* node,
                       size_t* input_count_return, InstructionOperand* inputs) {
  RiscvOperandGenerator g(selector);
  if (g.CanBeImmediate(node, *opcode_return)) {
    *opcode_return |= AddressingModeField::encode(kMode_MRI);
    inputs[0] = g.UseImmediate(node);
    *input_count_return = 1;
    return true;
  }
  return false;
}

static void VisitBinop(InstructionSelector* selector, Node* node,
                       InstructionCode opcode, bool has_reverse_opcode,
                       InstructionCode reverse_opcode,
                       FlagsContinuation* cont) {
  RiscvOperandGenerator g(selector);
  Int32BinopMatcher m(node);
  InstructionOperand inputs[2];
  size_t input_count = 0;
  InstructionOperand outputs[1];
  size_t output_count = 0;

  if (TryMatchImmediate(selector, &opcode, m.right().node(), &input_count,
                        &inputs[1])) {
    inputs[0] = g.UseRegisterOrImmediateZero(m.left().node());
    input_count++;
  } else if (has_reverse_opcode &&
             TryMatchImmediate(selector, &reverse_opcode, m.left().node(),
                               &input_count, &inputs[1])) {
    inputs[0] = g.UseRegisterOrImmediateZero(m.right().node());
    opcode = reverse_opcode;
    input_count++;
  } else {
    inputs[input_count++] = g.UseRegister(m.left().node());
    inputs[input_count++] = g.UseOperand(m.right().node(), opcode);
  }

  if (cont->IsDeoptimize()) {
    // If we can deoptimize as a result of the binop, we need to make sure that
    // the deopt inputs are not overwritten by the binop result. One way
    // to achieve that is to declare the output register as same-as-first.
    outputs[output_count++] = g.DefineSameAsFirst(node);
  } else {
    outputs[output_count++] = g.DefineAsRegister(node);
  }

  DCHECK_NE(0u, input_count);
  DCHECK_EQ(1u, output_count);
  DCHECK_GE(arraysize(inputs), input_count);
  DCHECK_GE(arraysize(outputs), output_count);

  selector->EmitWithContinuation(opcode, output_count, outputs, input_count,
                                 inputs, cont);
}

static void VisitBinop(InstructionSelector* selector, Node* node,
                       InstructionCode opcode, bool has_reverse_opcode,
                       InstructionCode reverse_opcode) {
  FlagsContinuation cont;
  VisitBinop(selector, node, opcode, has_reverse_opcode, reverse_opcode, &cont);
}

static void VisitBinop(InstructionSelector* selector, Node* node,
                       InstructionCode opcode, FlagsContinuation* cont) {
  VisitBinop(selector, node, opcode, false, kArchNop, cont);
}

static void VisitBinop(InstructionSelector* selector, Node* node,
                       InstructionCode opcode) {
  VisitBinop(selector, node, opcode, false, kArchNop);
}

void InstructionSelector::VisitStackSlot(Node* node) {
  StackSlotRepresentation rep = StackSlotRepresentationOf(node->op());
  int alignment = rep.alignment();
  int slot = frame_->AllocateSpillSlot(rep.size(), alignment);
  OperandGenerator g(this);

  Emit(kArchStackSlot, g.DefineAsRegister(node),
       sequence()->AddImmediate(Constant(slot)),
       sequence()->AddImmediate(Constant(alignment)), 0, nullptr);
}

void InstructionSelector::VisitAbortCSADcheck(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kArchAbortCSADcheck, g.NoOutput(), g.UseFixed(node->InputAt(0), a0));
}

void EmitLoad(InstructionSelector* selector, Node* node, InstructionCode opcode,
              Node* output = nullptr) {
  RiscvOperandGenerator g(selector);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);

  ExternalReferenceMatcher m(base);
  if (m.HasResolvedValue() && g.IsIntegerConstant(index) &&
      selector->CanAddressRelativeToRootsRegister(m.ResolvedValue())) {
    ptrdiff_t const delta =
        g.GetIntegerConstantValue(index) +
        TurboAssemblerBase::RootRegisterOffsetForExternalReference(
            selector->isolate(), m.ResolvedValue());
    // Check that the delta is a 32-bit integer due to the limitations of
    // immediate operands.
    if (is_int32(delta)) {
      opcode |= AddressingModeField::encode(kMode_Root);
      selector->Emit(opcode,
                     g.DefineAsRegister(output == nullptr ? node : output),
                     g.UseImmediate(static_cast<int32_t>(delta)));
      return;
    }
  }

  if (g.CanBeImmediate(index, opcode)) {
    selector->Emit(opcode | AddressingModeField::encode(kMode_MRI),
                   g.DefineAsRegister(output == nullptr ? node : output),
                   g.UseRegister(base), g.UseImmediate(index));
  } else {
    InstructionOperand addr_reg = g.TempRegister();
    selector->Emit(kRiscvAdd | AddressingModeField::encode(kMode_None),
                   addr_reg, g.UseRegister(index), g.UseRegister(base));
    // Emit desired load opcode, using temp addr_reg.
    selector->Emit(opcode | AddressingModeField::encode(kMode_MRI),
                   g.DefineAsRegister(output == nullptr ? node : output),
                   addr_reg, g.TempImmediate(0));
  }
}

void EmitS128Load(InstructionSelector* selector, Node* node,
                  InstructionCode opcode, VSew sew, Vlmul lmul) {
  RiscvOperandGenerator g(selector);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);

  if (g.CanBeImmediate(index, opcode)) {
    selector->Emit(opcode | AddressingModeField::encode(kMode_MRI),
                   g.DefineAsRegister(node), g.UseRegister(base),
                   g.UseImmediate(index), g.UseImmediate(sew),
                   g.UseImmediate(lmul));
  } else {
    InstructionOperand addr_reg = g.TempRegister();
    selector->Emit(kRiscvAdd | AddressingModeField::encode(kMode_None),
                   addr_reg, g.UseRegister(index), g.UseRegister(base));
    // Emit desired load opcode, using temp addr_reg.
    selector->Emit(opcode | AddressingModeField::encode(kMode_MRI),
                   g.DefineAsRegister(node), addr_reg, g.TempImmediate(0),
                   g.UseImmediate(sew), g.UseImmediate(lmul));
  }
}

void InstructionSelector::VisitStoreLane(Node* node) {
  StoreLaneParameters params = StoreLaneParametersOf(node->op());
  LoadStoreLaneParams f(params.rep, params.laneidx);
  InstructionCode opcode = kRiscvS128StoreLane;
  opcode |= MiscField::encode(f.sz);

  RiscvOperandGenerator g(this);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);
  InstructionOperand addr_reg = g.TempRegister();
  Emit(kRiscvAdd, addr_reg, g.UseRegister(base), g.UseRegister(index));
  InstructionOperand inputs[4] = {
      g.UseRegister(node->InputAt(2)),
      g.UseImmediate(f.laneidx),
      addr_reg,
      g.TempImmediate(0),
  };
  opcode |= AddressingModeField::encode(kMode_MRI);
  Emit(opcode, 0, nullptr, 4, inputs);
}

void InstructionSelector::VisitLoadLane(Node* node) {
  LoadLaneParameters params = LoadLaneParametersOf(node->op());
  LoadStoreLaneParams f(params.rep.representation(), params.laneidx);
  InstructionCode opcode = kRiscvS128LoadLane;
  opcode |= MiscField::encode(f.sz);

  RiscvOperandGenerator g(this);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);
  InstructionOperand addr_reg = g.TempRegister();
  Emit(kRiscvAdd, addr_reg, g.UseRegister(base), g.UseRegister(index));
  opcode |= AddressingModeField::encode(kMode_MRI);
  Emit(opcode, g.DefineSameAsFirst(node), g.UseRegister(node->InputAt(2)),
       g.UseImmediate(params.laneidx), addr_reg, g.TempImmediate(0));
}

void InstructionSelector::VisitLoadTransform(Node* node) {
  LoadTransformParameters params = LoadTransformParametersOf(node->op());

  switch (params.transformation) {
    case LoadTransformation::kS128Load8Splat:
      EmitS128Load(this, node, kRiscvS128LoadSplat, E8, m1);
      break;
    case LoadTransformation::kS128Load16Splat:
      EmitS128Load(this, node, kRiscvS128LoadSplat, E16, m1);
      break;
    case LoadTransformation::kS128Load32Splat:
      EmitS128Load(this, node, kRiscvS128LoadSplat, E32, m1);
      break;
    case LoadTransformation::kS128Load64Splat:
      EmitS128Load(this, node, kRiscvS128LoadSplat, E64, m1);
      break;
    case LoadTransformation::kS128Load8x8S:
      EmitS128Load(this, node, kRiscvS128Load64ExtendS, E16, m1);
      break;
    case LoadTransformation::kS128Load8x8U:
      EmitS128Load(this, node, kRiscvS128Load64ExtendU, E16, m1);
      break;
    case LoadTransformation::kS128Load16x4S:
      EmitS128Load(this, node, kRiscvS128Load64ExtendS, E32, m1);
      break;
    case LoadTransformation::kS128Load16x4U:
      EmitS128Load(this, node, kRiscvS128Load64ExtendU, E32, m1);
      break;
    case LoadTransformation::kS128Load32x2S:
      EmitS128Load(this, node, kRiscvS128Load64ExtendS, E64, m1);
      break;
    case LoadTransformation::kS128Load32x2U:
      EmitS128Load(this, node, kRiscvS128Load64ExtendU, E64, m1);
      break;
    case LoadTransformation::kS128Load32Zero:
      EmitS128Load(this, node, kRiscvS128Load32Zero, E32, m1);
      break;
    case LoadTransformation::kS128Load64Zero:
      EmitS128Load(this, node, kRiscvS128Load64Zero, E64, m1);
      break;
    default:
      UNIMPLEMENTED();
  }
}

void InstructionSelector::VisitLoad(Node* node) {
  LoadRepresentation load_rep = LoadRepresentationOf(node->op());

  InstructionCode opcode = kArchNop;
  switch (load_rep.representation()) {
    case MachineRepresentation::kFloat32:
      opcode = kRiscvLoadFloat;
      break;
    case MachineRepresentation::kFloat64:
      opcode = kRiscvLoadDouble;
      break;
    case MachineRepresentation::kBit:  // Fall through.
    case MachineRepresentation::kWord8:
      opcode = load_rep.IsUnsigned() ? kRiscvLbu : kRiscvLb;
      break;
    case MachineRepresentation::kWord16:
      opcode = load_rep.IsUnsigned() ? kRiscvLhu : kRiscvLh;
      break;
    case MachineRepresentation::kTaggedSigned:   // Fall through.
    case MachineRepresentation::kTaggedPointer:  // Fall through.
    case MachineRepresentation::kTagged:         // Fall through.
    case MachineRepresentation::kWord32:
      opcode = kRiscvLw;
      break;
    case MachineRepresentation::kSimd128:
      opcode = kRiscvRvvLd;
      break;
    case MachineRepresentation::kCompressedPointer:
    case MachineRepresentation::kCompressed:
    case MachineRepresentation::kSandboxedPointer:
    case MachineRepresentation::kMapWord:  // Fall through.
    case MachineRepresentation::kWord64:
    case MachineRepresentation::kNone:

      UNREACHABLE();
  }

  EmitLoad(this, node, opcode);
}

void InstructionSelector::VisitProtectedLoad(Node* node) {
  // TODO(eholk)
  UNIMPLEMENTED();
}

void InstructionSelector::VisitStore(Node* node) {
  RiscvOperandGenerator g(this);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);
  Node* value = node->InputAt(2);

  StoreRepresentation store_rep = StoreRepresentationOf(node->op());
  WriteBarrierKind write_barrier_kind = store_rep.write_barrier_kind();
  MachineRepresentation rep = store_rep.representation();

  // TODO(riscv): I guess this could be done in a better way.
  if (write_barrier_kind != kNoWriteBarrier &&
      V8_LIKELY(!FLAG_disable_write_barriers)) {
    DCHECK(CanBeTaggedPointer(rep));
    InstructionOperand inputs[3];
    size_t input_count = 0;
    inputs[input_count++] = g.UseUniqueRegister(base);
    inputs[input_count++] = g.UseUniqueRegister(index);
    inputs[input_count++] = g.UseUniqueRegister(value);
    RecordWriteMode record_write_mode =
        WriteBarrierKindToRecordWriteMode(write_barrier_kind);
    InstructionOperand temps[] = {g.TempRegister(), g.TempRegister()};
    size_t const temp_count = arraysize(temps);
    InstructionCode code = kArchStoreWithWriteBarrier;
    code |= MiscField::encode(static_cast<int>(record_write_mode));
    Emit(code, 0, nullptr, input_count, inputs, temp_count, temps);
  } else {
    ArchOpcode opcode;
    switch (rep) {
      case MachineRepresentation::kFloat32:
        opcode = kRiscvStoreFloat;
        break;
      case MachineRepresentation::kFloat64:
        opcode = kRiscvStoreDouble;
        break;
      case MachineRepresentation::kBit:  // Fall through.
      case MachineRepresentation::kWord8:
        opcode = kRiscvSb;
        break;
      case MachineRepresentation::kWord16:
        opcode = kRiscvSh;
        break;
      case MachineRepresentation::kTaggedSigned:   // Fall through.
      case MachineRepresentation::kTaggedPointer:  // Fall through.
      case MachineRepresentation::kTagged:
      case MachineRepresentation::kWord32:
        opcode = kRiscvSw;
        break;
      case MachineRepresentation::kSimd128:
        opcode = kRiscvRvvSt;
        break;
      case MachineRepresentation::kCompressedPointer:  // Fall through.
      case MachineRepresentation::kCompressed:
        UNREACHABLE();
      case MachineRepresentation::kSandboxedPointer:
      case MachineRepresentation::kMapWord:  // Fall through.
      case MachineRepresentation::kNone:
      case MachineRepresentation::kWord64:
        UNREACHABLE();
    }

    if (g.CanBeImmediate(index, opcode)) {
      Emit(opcode | AddressingModeField::encode(kMode_MRI), g.NoOutput(),
           g.UseRegister(base), g.UseImmediate(index),
           g.UseRegisterOrImmediateZero(value));
    } else {
      InstructionOperand addr_reg = g.TempRegister();
      Emit(kRiscvAdd | AddressingModeField::encode(kMode_None), addr_reg,
           g.UseRegister(index), g.UseRegister(base));
      // Emit desired store opcode, using temp addr_reg.
      Emit(opcode | AddressingModeField::encode(kMode_MRI), g.NoOutput(),
           addr_reg, g.TempImmediate(0), g.UseRegisterOrImmediateZero(value));
    }
  }
}

void InstructionSelector::VisitProtectedStore(Node* node) {
  // TODO(eholk)
  UNIMPLEMENTED();
}

void InstructionSelector::VisitWord32And(Node* node) {
  VisitBinop(this, node, kRiscvAnd, true, kRiscvAnd);
}

void InstructionSelector::VisitWord32Or(Node* node) {
  VisitBinop(this, node, kRiscvOr, true, kRiscvOr);
}

void InstructionSelector::VisitWord32Xor(Node* node) {
  Int32BinopMatcher m(node);
  if (m.left().IsWord32Or() && CanCover(node, m.left().node()) &&
      m.right().Is(-1)) {
    Int32BinopMatcher mleft(m.left().node());
    if (!mleft.right().HasResolvedValue()) {
      RiscvOperandGenerator g(this);
      Emit(kRiscvNor, g.DefineAsRegister(node),
           g.UseRegister(mleft.left().node()),
           g.UseRegister(mleft.right().node()));
      return;
    }
  }
  if (m.right().Is(-1)) {
    // Use Nor for bit negation and eliminate constant loading for xori.
    RiscvOperandGenerator g(this);
    Emit(kRiscvNor, g.DefineAsRegister(node), g.UseRegister(m.left().node()),
         g.TempImmediate(0));
    return;
  }
  VisitBinop(this, node, kRiscvXor, true, kRiscvXor);
}

void InstructionSelector::VisitWord32Shl(Node* node) {
  Int32BinopMatcher m(node);
  if (m.left().IsWord32And() && CanCover(node, m.left().node()) &&
      m.right().IsInRange(1, 31)) {
    RiscvOperandGenerator g(this);
    Int32BinopMatcher mleft(m.left().node());
    // Match Word32Shl(Word32And(x, mask), imm) to Shl where the mask is
    // contiguous, and the shift immediate non-zero.
    if (mleft.right().HasResolvedValue()) {
      uint32_t mask = mleft.right().ResolvedValue();
      uint32_t mask_width = base::bits::CountPopulation(mask);
      uint32_t mask_msb = base::bits::CountLeadingZeros32(mask);
      if ((mask_width != 0) && (mask_msb + mask_width == 32)) {
        uint32_t shift = m.right().ResolvedValue();
        DCHECK_EQ(0u, base::bits::CountTrailingZeros32(mask));
        DCHECK_NE(0u, shift);
        if ((shift + mask_width) >= 32) {
          // If the mask is contiguous and reaches or extends beyond the top
          // bit, only the shift is needed.
          Emit(kRiscvShl32, g.DefineAsRegister(node),
               g.UseRegister(mleft.left().node()),
               g.UseImmediate(m.right().node()));
          return;
        }
      }
    }
  }
  VisitRRO(this, kRiscvShl32, node);
}

void InstructionSelector::VisitWord32Shr(Node* node) {
  VisitRRO(this, kRiscvShr32, node);
}

void InstructionSelector::VisitWord32Sar(Node* node) {
  Int32BinopMatcher m(node);
  if (m.left().IsWord32Shl() && CanCover(node, m.left().node())) {
    Int32BinopMatcher mleft(m.left().node());
    if (m.right().HasResolvedValue() && mleft.right().HasResolvedValue()) {
      RiscvOperandGenerator g(this);
      uint32_t sar = m.right().ResolvedValue();
      uint32_t shl = mleft.right().ResolvedValue();
      if ((sar == shl) && (sar == 16)) {
        Emit(kRiscvSignExtendShort, g.DefineAsRegister(node),
             g.UseRegister(mleft.left().node()));
        return;
      } else if ((sar == shl) && (sar == 24)) {
        Emit(kRiscvSignExtendByte, g.DefineAsRegister(node),
             g.UseRegister(mleft.left().node()));
        return;
      } else if ((sar == shl) && (sar == 32)) {
        Emit(kRiscvShl32, g.DefineAsRegister(node),
             g.UseRegister(mleft.left().node()), g.TempImmediate(0));
        return;
      }
    }
  }
  VisitRRO(this, kRiscvSar32, node);
}

void InstructionSelector::VisitWord32Rol(Node* node) { UNREACHABLE(); }

void InstructionSelector::VisitWord32Ror(Node* node) {
  VisitRRO(this, kRiscvRor32, node);
}

void InstructionSelector::VisitWord32Clz(Node* node) {
  VisitRR(this, kRiscvClz32, node);
}

void InstructionSelector::VisitWord32ReverseBits(Node* node) { UNREACHABLE(); }

void InstructionSelector::VisitWord64ReverseBytes(Node* node) { UNREACHABLE(); }

void InstructionSelector::VisitWord32ReverseBytes(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvByteSwap32, g.DefineAsRegister(node),
       g.UseRegister(node->InputAt(0)));
}

void InstructionSelector::VisitSimd128ReverseBytes(Node* node) {
  UNREACHABLE();
}

void InstructionSelector::VisitWord32Ctz(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvCtz32, g.DefineAsRegister(node), g.UseRegister(node->InputAt(0)));
}

void InstructionSelector::VisitWord32Popcnt(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvPopcnt32, g.DefineAsRegister(node),
       g.UseRegister(node->InputAt(0)));
}

void InstructionSelector::VisitInt32Add(Node* node) {
  VisitBinop(this, node, kRiscvAdd, true, kRiscvAdd);
}

void InstructionSelector::VisitInt32Sub(Node* node) {
  VisitBinop(this, node, kRiscvSub);
}

void InstructionSelector::VisitInt32Mul(Node* node) {
  RiscvOperandGenerator g(this);
  Int32BinopMatcher m(node);
  if (m.right().HasResolvedValue() && m.right().ResolvedValue() > 0) {
    uint32_t value = static_cast<uint32_t>(m.right().ResolvedValue());
    if (base::bits::IsPowerOfTwo(value)) {
      Emit(kRiscvShl32 | AddressingModeField::encode(kMode_None),
           g.DefineAsRegister(node), g.UseRegister(m.left().node()),
           g.TempImmediate(base::bits::WhichPowerOfTwo(value)));
      return;
    }
    if (base::bits::IsPowerOfTwo(value + 1)) {
      InstructionOperand temp = g.TempRegister();
      Emit(kRiscvShl32 | AddressingModeField::encode(kMode_None), temp,
           g.UseRegister(m.left().node()),
           g.TempImmediate(base::bits::WhichPowerOfTwo(value + 1)));
      Emit(kRiscvSub | AddressingModeField::encode(kMode_None),
           g.DefineAsRegister(node), temp, g.UseRegister(m.left().node()));
      return;
    }
  }

  VisitRRR(this, kRiscvMul32, node);
}

void InstructionSelector::VisitI32x4ExtAddPairwiseI16x8S(Node* node) {
  RiscvOperandGenerator g(this);
  InstructionOperand src1 = g.TempSimd128Register();
  InstructionOperand src2 = g.TempSimd128Register();
  InstructionOperand src = g.UseUniqueRegister(node->InputAt(0));
  Emit(kRiscvVrgather, src1, src, g.UseImmediate64(0x0006000400020000),
       g.UseImmediate(int8_t(E16)), g.UseImmediate(int8_t(m1)));
  Emit(kRiscvVrgather, src2, src, g.UseImmediate64(0x0007000500030001),
       g.UseImmediate(int8_t(E16)), g.UseImmediate(int8_t(m1)));
  Emit(kRiscvVwadd, g.DefineAsRegister(node), src1, src2,
       g.UseImmediate(int8_t(E16)), g.UseImmediate(int8_t(mf2)));
}

void InstructionSelector::VisitI32x4ExtAddPairwiseI16x8U(Node* node) {
  RiscvOperandGenerator g(this);
  InstructionOperand src1 = g.TempSimd128Register();
  InstructionOperand src2 = g.TempSimd128Register();
  InstructionOperand src = g.UseUniqueRegister(node->InputAt(0));
  Emit(kRiscvVrgather, src1, src, g.UseImmediate64(0x0006000400020000),
       g.UseImmediate(int8_t(E16)), g.UseImmediate(int8_t(m1)));
  Emit(kRiscvVrgather, src2, src, g.UseImmediate64(0x0007000500030001),
       g.UseImmediate(int8_t(E16)), g.UseImmediate(int8_t(m1)));
  Emit(kRiscvVwaddu, g.DefineAsRegister(node), src1, src2,
       g.UseImmediate(int8_t(E16)), g.UseImmediate(int8_t(mf2)));
}

void InstructionSelector::VisitI16x8ExtAddPairwiseI8x16S(Node* node) {
  RiscvOperandGenerator g(this);
  InstructionOperand src1 = g.TempSimd128Register();
  InstructionOperand src2 = g.TempSimd128Register();
  InstructionOperand src = g.UseUniqueRegister(node->InputAt(0));
  Emit(kRiscvVrgather, src1, src, g.UseImmediate64(0x0E0C0A0806040200),
       g.UseImmediate(int8_t(E8)), g.UseImmediate(int8_t(m1)));
  Emit(kRiscvVrgather, src2, src, g.UseImmediate64(0x0F0D0B0907050301),
       g.UseImmediate(int8_t(E8)), g.UseImmediate(int8_t(m1)));
  Emit(kRiscvVwadd, g.DefineAsRegister(node), src1, src2,
       g.UseImmediate(int8_t(E8)), g.UseImmediate(int8_t(mf2)));
}

void InstructionSelector::VisitI16x8ExtAddPairwiseI8x16U(Node* node) {
  RiscvOperandGenerator g(this);
  InstructionOperand src1 = g.TempSimd128Register();
  InstructionOperand src2 = g.TempSimd128Register();
  InstructionOperand src = g.UseUniqueRegister(node->InputAt(0));
  Emit(kRiscvVrgather, src1, src, g.UseImmediate64(0x0E0C0A0806040200),
       g.UseImmediate(int8_t(E8)), g.UseImmediate(int8_t(m1)));
  Emit(kRiscvVrgather, src2, src, g.UseImmediate64(0x0F0D0B0907050301),
       g.UseImmediate(int8_t(E8)), g.UseImmediate(int8_t(m1)));
  Emit(kRiscvVwaddu, g.DefineAsRegister(node), src1, src2,
       g.UseImmediate(int8_t(E8)), g.UseImmediate(int8_t(mf2)));
}

void InstructionSelector::VisitInt32MulHigh(Node* node) {
  VisitRRR(this, kRiscvMulHigh32, node);
}

void InstructionSelector::VisitUint32MulHigh(Node* node) {
  VisitRRR(this, kRiscvMulHighU32, node);
}

void InstructionSelector::VisitInt32Div(Node* node) {
  RiscvOperandGenerator g(this);
  Int32BinopMatcher m(node);
  Emit(kRiscvDiv32, g.DefineSameAsFirst(node), g.UseRegister(m.left().node()),
       g.UseRegister(m.right().node()));
}

void InstructionSelector::VisitUint32Div(Node* node) {
  RiscvOperandGenerator g(this);
  Int32BinopMatcher m(node);
  Emit(kRiscvDivU32, g.DefineSameAsFirst(node), g.UseRegister(m.left().node()),
       g.UseRegister(m.right().node()));
}

void InstructionSelector::VisitInt32Mod(Node* node) {
  RiscvOperandGenerator g(this);
  Int32BinopMatcher m(node);
  Emit(kRiscvMod32, g.DefineAsRegister(node), g.UseRegister(m.left().node()),
       g.UseRegister(m.right().node()));
}

void InstructionSelector::VisitUint32Mod(Node* node) {
  RiscvOperandGenerator g(this);
  Int32BinopMatcher m(node);
  Emit(kRiscvModU32, g.DefineAsRegister(node), g.UseRegister(m.left().node()),
       g.UseRegister(m.right().node()));
}

void InstructionSelector::VisitChangeFloat32ToFloat64(Node* node) {
  VisitRR(this, kRiscvCvtDS, node);
}

void InstructionSelector::VisitRoundInt32ToFloat32(Node* node) {
  VisitRR(this, kRiscvCvtSW, node);
}

void InstructionSelector::VisitRoundUint32ToFloat32(Node* node) {
  VisitRR(this, kRiscvCvtSUw, node);
}

void InstructionSelector::VisitChangeInt32ToFloat64(Node* node) {
  VisitRR(this, kRiscvCvtDW, node);
}

void InstructionSelector::VisitChangeUint32ToFloat64(Node* node) {
  VisitRR(this, kRiscvCvtDUw, node);
}

void InstructionSelector::VisitTruncateFloat32ToInt32(Node* node) {
  RiscvOperandGenerator g(this);
  InstructionCode opcode = kRiscvTruncWS;
  TruncateKind kind = OpParameter<TruncateKind>(node->op());
  if (kind == TruncateKind::kSetOverflowToMin) {
    opcode |= MiscField::encode(true);
  }
  Emit(opcode, g.DefineAsRegister(node), g.UseRegister(node->InputAt(0)));
}

void InstructionSelector::VisitTruncateFloat32ToUint32(Node* node) {
  RiscvOperandGenerator g(this);
  InstructionCode opcode = kRiscvTruncUwS;
  TruncateKind kind = OpParameter<TruncateKind>(node->op());
  if (kind == TruncateKind::kSetOverflowToMin) {
    opcode |= MiscField::encode(true);
  }
  Emit(opcode, g.DefineAsRegister(node), g.UseRegister(node->InputAt(0)));
}

void InstructionSelector::VisitChangeFloat64ToInt32(Node* node) {
  RiscvOperandGenerator g(this);
  Node* value = node->InputAt(0);
  if (CanCover(node, value)) {
    if (value->opcode() == IrOpcode::kChangeFloat32ToFloat64) {
      // Match float32 -> float64 -> int32 representation change path.
      Emit(kRiscvTruncWS, g.DefineAsRegister(node),
           g.UseRegister(value->InputAt(0)));
      return;
    }
  }
  VisitRR(this, kRiscvTruncWD, node);
}

void InstructionSelector::VisitChangeFloat64ToUint32(Node* node) {
  VisitRR(this, kRiscvTruncUwD, node);
}

void InstructionSelector::VisitTruncateFloat64ToUint32(Node* node) {
  VisitRR(this, kRiscvTruncUwD, node);
}

void InstructionSelector::VisitTruncateFloat64ToFloat32(Node* node) {
  RiscvOperandGenerator g(this);
  Node* value = node->InputAt(0);
  // Match TruncateFloat64ToFloat32(ChangeInt32ToFloat64) to corresponding
  // instruction.
  if (CanCover(node, value) &&
      value->opcode() == IrOpcode::kChangeInt32ToFloat64) {
    Emit(kRiscvCvtSW, g.DefineAsRegister(node),
         g.UseRegister(value->InputAt(0)));
    return;
  }
  VisitRR(this, kRiscvCvtSD, node);
}

void InstructionSelector::VisitTruncateFloat64ToWord32(Node* node) {
  VisitRR(this, kArchTruncateDoubleToI, node);
}

void InstructionSelector::VisitRoundFloat64ToInt32(Node* node) {
  VisitRR(this, kRiscvTruncWD, node);
}

void InstructionSelector::VisitBitcastFloat32ToInt32(Node* node) {
  VisitRR(this, kRiscvBitcastFloat32ToInt32, node);
}

void InstructionSelector::VisitBitcastInt32ToFloat32(Node* node) {
  VisitRR(this, kRiscvBitcastInt32ToFloat32, node);
}

void InstructionSelector::VisitFloat32Add(Node* node) {
  VisitRRR(this, kRiscvAddS, node);
}

void InstructionSelector::VisitFloat64Add(Node* node) {
  VisitRRR(this, kRiscvAddD, node);
}

void InstructionSelector::VisitFloat32Sub(Node* node) {
  VisitRRR(this, kRiscvSubS, node);
}

void InstructionSelector::VisitFloat64Sub(Node* node) {
  VisitRRR(this, kRiscvSubD, node);
}

void InstructionSelector::VisitFloat32Mul(Node* node) {
  VisitRRR(this, kRiscvMulS, node);
}

void InstructionSelector::VisitFloat64Mul(Node* node) {
  VisitRRR(this, kRiscvMulD, node);
}

void InstructionSelector::VisitFloat32Div(Node* node) {
  VisitRRR(this, kRiscvDivS, node);
}

void InstructionSelector::VisitFloat64Div(Node* node) {
  VisitRRR(this, kRiscvDivD, node);
}

void InstructionSelector::VisitFloat64Mod(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvModD, g.DefineAsFixed(node, fa0),
       g.UseFixed(node->InputAt(0), fa0), g.UseFixed(node->InputAt(1), fa1))
      ->MarkAsCall();
}

void InstructionSelector::VisitFloat32Max(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvFloat32Max, g.DefineAsRegister(node),
       g.UseRegister(node->InputAt(0)), g.UseRegister(node->InputAt(1)));
}

void InstructionSelector::VisitFloat64Max(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvFloat64Max, g.DefineAsRegister(node),
       g.UseRegister(node->InputAt(0)), g.UseRegister(node->InputAt(1)));
}

void InstructionSelector::VisitFloat32Min(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvFloat32Min, g.DefineAsRegister(node),
       g.UseRegister(node->InputAt(0)), g.UseRegister(node->InputAt(1)));
}

void InstructionSelector::VisitFloat64Min(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvFloat64Min, g.DefineAsRegister(node),
       g.UseRegister(node->InputAt(0)), g.UseRegister(node->InputAt(1)));
}

void InstructionSelector::VisitFloat32Abs(Node* node) {
  VisitRR(this, kRiscvAbsS, node);
}

void InstructionSelector::VisitFloat64Abs(Node* node) {
  VisitRR(this, kRiscvAbsD, node);
}

void InstructionSelector::VisitFloat32Sqrt(Node* node) {
  VisitRR(this, kRiscvSqrtS, node);
}

void InstructionSelector::VisitFloat64Sqrt(Node* node) {
  VisitRR(this, kRiscvSqrtD, node);
}

void InstructionSelector::VisitFloat32RoundDown(Node* node) {
  VisitRR(this, kRiscvFloat32RoundDown, node);
}

void InstructionSelector::VisitFloat64RoundDown(Node* node) { UNIMPLEMENTED(); }

void InstructionSelector::VisitFloat32RoundUp(Node* node) {
  VisitRR(this, kRiscvFloat32RoundUp, node);
}

void InstructionSelector::VisitFloat64RoundUp(Node* node) { UNIMPLEMENTED(); }

void InstructionSelector::VisitFloat32RoundTruncate(Node* node) {
  VisitRR(this, kRiscvFloat32RoundTruncate, node);
}

void InstructionSelector::VisitFloat64RoundTruncate(Node* node) {
  UNIMPLEMENTED();
}

void InstructionSelector::VisitFloat64RoundTiesAway(Node* node) {
  UNREACHABLE();
}

void InstructionSelector::VisitFloat32RoundTiesEven(Node* node) {
  VisitRR(this, kRiscvFloat32RoundTiesEven, node);
}

void InstructionSelector::VisitFloat64RoundTiesEven(Node* node) {
  UNIMPLEMENTED();
}

void InstructionSelector::VisitFloat32Neg(Node* node) {
  VisitRR(this, kRiscvNegS, node);
}

void InstructionSelector::VisitFloat64Neg(Node* node) {
  VisitRR(this, kRiscvNegD, node);
}

void InstructionSelector::VisitFloat64Ieee754Binop(Node* node,
                                                   InstructionCode opcode) {
  RiscvOperandGenerator g(this);
  Emit(opcode, g.DefineAsFixed(node, fa0), g.UseFixed(node->InputAt(0), fa0),
       g.UseFixed(node->InputAt(1), fa1))
      ->MarkAsCall();
}

void InstructionSelector::VisitFloat64Ieee754Unop(Node* node,
                                                  InstructionCode opcode) {
  RiscvOperandGenerator g(this);
  Emit(opcode, g.DefineAsFixed(node, fa0), g.UseFixed(node->InputAt(0), fa1))
      ->MarkAsCall();
}

void InstructionSelector::EmitPrepareArguments(
    ZoneVector<PushParameter>* arguments, const CallDescriptor* call_descriptor,
    Node* node) {
  RiscvOperandGenerator g(this);

  // Prepare for C function call.
  if (call_descriptor->IsCFunctionCall()) {
    Emit(kArchPrepareCallCFunction | MiscField::encode(static_cast<int>(
                                         call_descriptor->ParameterCount())),
         0, nullptr, 0, nullptr);

    // Poke any stack arguments.
    int slot = kCArgSlotCount;
    for (PushParameter input : (*arguments)) {
      Emit(kRiscvStoreToStackSlot, g.NoOutput(), g.UseRegister(input.node),
           g.TempImmediate(slot << kSystemPointerSizeLog2));
      ++slot;
    }
  } else {
    int push_count = static_cast<int>(call_descriptor->ParameterSlotCount());
    if (push_count > 0) {
      Emit(kRiscvStackClaim, g.NoOutput(),
           g.TempImmediate(arguments->size() << kSystemPointerSizeLog2));
    }
    for (size_t n = 0; n < arguments->size(); ++n) {
      PushParameter input = (*arguments)[n];
      if (input.node) {
        Emit(kRiscvStoreToStackSlot, g.NoOutput(), g.UseRegister(input.node),
             g.TempImmediate(static_cast<int>(n << kSystemPointerSizeLog2)));
      }
    }
  }
}

void InstructionSelector::EmitPrepareResults(
    ZoneVector<PushParameter>* results, const CallDescriptor* call_descriptor,
    Node* node) {
  RiscvOperandGenerator g(this);

  int reverse_slot = 1;
  for (PushParameter output : *results) {
    if (!output.location.IsCallerFrameSlot()) continue;
    // Skip any alignment holes in nodes.
    if (output.node != nullptr) {
      DCHECK(!call_descriptor->IsCFunctionCall());
      if (output.location.GetType() == MachineType::Float32()) {
        MarkAsFloat32(output.node);
      } else if (output.location.GetType() == MachineType::Float64()) {
        MarkAsFloat64(output.node);
      }
      Emit(kRiscvPeek, g.DefineAsRegister(output.node),
           g.UseImmediate(reverse_slot));
    }
    reverse_slot += output.location.GetSizeInPointers();
  }
}

bool InstructionSelector::IsTailCallAddressImmediate() { return false; }

void InstructionSelector::VisitUnalignedLoad(Node* node) {
  LoadRepresentation load_rep = LoadRepresentationOf(node->op());
  RiscvOperandGenerator g(this);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);

  ArchOpcode opcode;
  switch (load_rep.representation()) {
    case MachineRepresentation::kFloat32:
      opcode = kRiscvULoadFloat;
      break;
    case MachineRepresentation::kFloat64:
      opcode = kRiscvULoadDouble;
      break;
    case MachineRepresentation::kWord8:
      opcode = load_rep.IsUnsigned() ? kRiscvLbu : kRiscvLb;
      break;
    case MachineRepresentation::kWord16:
      opcode = load_rep.IsUnsigned() ? kRiscvUlhu : kRiscvUlh;
      break;
    case MachineRepresentation::kTaggedSigned:   // Fall through.
    case MachineRepresentation::kTaggedPointer:  // Fall through.
    case MachineRepresentation::kTagged:         // Fall through.
    case MachineRepresentation::kWord32:
      opcode = kRiscvUlw;
      break;
    case MachineRepresentation::kSimd128:
      opcode = kRiscvRvvLd;
      break;
    case MachineRepresentation::kBit:                // Fall through.
    case MachineRepresentation::kCompressedPointer:  // Fall through.
    case MachineRepresentation::kCompressed:         // Fall through.
    case MachineRepresentation::kSandboxedPointer:   // Fall through.
    case MachineRepresentation::kMapWord:            // Fall through.
    case MachineRepresentation::kWord64:
    case MachineRepresentation::kNone:
      UNREACHABLE();
  }

  if (g.CanBeImmediate(index, opcode)) {
    Emit(opcode | AddressingModeField::encode(kMode_MRI),
         g.DefineAsRegister(node), g.UseRegister(base), g.UseImmediate(index));
  } else {
    InstructionOperand addr_reg = g.TempRegister();
    Emit(kRiscvAdd | AddressingModeField::encode(kMode_None), addr_reg,
         g.UseRegister(index), g.UseRegister(base));
    // Emit desired load opcode, using temp addr_reg.
    Emit(opcode | AddressingModeField::encode(kMode_MRI),
         g.DefineAsRegister(node), addr_reg, g.TempImmediate(0));
  }
}

void InstructionSelector::VisitUnalignedStore(Node* node) {
  RiscvOperandGenerator g(this);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);
  Node* value = node->InputAt(2);

  UnalignedStoreRepresentation rep = UnalignedStoreRepresentationOf(node->op());
  ArchOpcode opcode;
  switch (rep) {
    case MachineRepresentation::kFloat32:
      opcode = kRiscvUStoreFloat;
      break;
    case MachineRepresentation::kFloat64:
      opcode = kRiscvUStoreDouble;
      break;
    case MachineRepresentation::kWord8:
      opcode = kRiscvSb;
      break;
    case MachineRepresentation::kWord16:
      opcode = kRiscvUsh;
      break;
    case MachineRepresentation::kTaggedSigned:   // Fall through.
    case MachineRepresentation::kTaggedPointer:  // Fall through.
    case MachineRepresentation::kTagged:         // Fall through.
    case MachineRepresentation::kWord32:
      opcode = kRiscvUsw;
      break;
    case MachineRepresentation::kSimd128:
      opcode = kRiscvRvvSt;
      break;
    case MachineRepresentation::kBit:                // Fall through.
    case MachineRepresentation::kCompressedPointer:  // Fall through.
    case MachineRepresentation::kCompressed:         // Fall through.
    case MachineRepresentation::kSandboxedPointer:
    case MachineRepresentation::kMapWord:  // Fall through.
    case MachineRepresentation::kNone:
    case MachineRepresentation::kWord64:
      UNREACHABLE();
  }

  if (g.CanBeImmediate(index, opcode)) {
    Emit(opcode | AddressingModeField::encode(kMode_MRI), g.NoOutput(),
         g.UseRegister(base), g.UseImmediate(index),
         g.UseRegisterOrImmediateZero(value));
  } else {
    InstructionOperand addr_reg = g.TempRegister();
    Emit(kRiscvAdd | AddressingModeField::encode(kMode_None), addr_reg,
         g.UseRegister(index), g.UseRegister(base));
    // Emit desired store opcode, using temp addr_reg.
    Emit(opcode | AddressingModeField::encode(kMode_MRI), g.NoOutput(),
         addr_reg, g.TempImmediate(0), g.UseRegisterOrImmediateZero(value));
  }
}

namespace {

// Shared routine for multiple compare operations.
static void VisitCompare(InstructionSelector* selector, InstructionCode opcode,
                         InstructionOperand left, InstructionOperand right,
                         FlagsContinuation* cont) {
  selector->EmitWithContinuation(opcode, left, right, cont);
}

// Shared routine for multiple compare operations.
static void VisitWordCompareZero(InstructionSelector* selector,
                                 InstructionOperand value,
                                 FlagsContinuation* cont) {
  selector->EmitWithContinuation(kRiscvCmpZero, value, cont);
}

// Shared routine for multiple float32 compare operations.
void VisitFloat32Compare(InstructionSelector* selector, Node* node,
                         FlagsContinuation* cont) {
  RiscvOperandGenerator g(selector);
  Float32BinopMatcher m(node);
  InstructionOperand lhs, rhs;

  lhs = m.left().IsZero() ? g.UseImmediate(m.left().node())
                          : g.UseRegister(m.left().node());
  rhs = m.right().IsZero() ? g.UseImmediate(m.right().node())
                           : g.UseRegister(m.right().node());
  VisitCompare(selector, kRiscvCmpS, lhs, rhs, cont);
}

// Shared routine for multiple float64 compare operations.
void VisitFloat64Compare(InstructionSelector* selector, Node* node,
                         FlagsContinuation* cont) {
  RiscvOperandGenerator g(selector);
  Float64BinopMatcher m(node);
  InstructionOperand lhs, rhs;

  lhs = m.left().IsZero() ? g.UseImmediate(m.left().node())
                          : g.UseRegister(m.left().node());
  rhs = m.right().IsZero() ? g.UseImmediate(m.right().node())
                           : g.UseRegister(m.right().node());
  VisitCompare(selector, kRiscvCmpD, lhs, rhs, cont);
}

// Shared routine for multiple word compare operations.
void VisitWordCompare(InstructionSelector* selector, Node* node,
                      InstructionCode opcode, FlagsContinuation* cont,
                      bool commutative) {
  RiscvOperandGenerator g(selector);
  Node* left = node->InputAt(0);
  Node* right = node->InputAt(1);
  // If one of the two inputs is an immediate, make sure it's on the right.
  if (!g.CanBeImmediate(right, opcode) && g.CanBeImmediate(left, opcode)) {
    cont->Commute();
    std::swap(left, right);
  }
  // Match immediates on right side of comparison.
  if (g.CanBeImmediate(right, opcode)) {
    if (opcode == kRiscvTst) {
      VisitCompare(selector, opcode, g.UseRegister(left), g.UseImmediate(right),
                   cont);
    } else {
      switch (cont->condition()) {
        case kEqual:
        case kNotEqual:
          if (cont->IsSet()) {
            VisitCompare(selector, opcode, g.UseRegister(left),
                         g.UseImmediate(right), cont);
          } else {
            Int32BinopMatcher m(node, true);
            NumberBinopMatcher n(node, true);
            if (m.right().Is(0) || n.right().IsZero()) {
              VisitWordCompareZero(selector, g.UseRegisterOrImmediateZero(left),
                                   cont);
            } else {
              VisitCompare(selector, opcode, g.UseRegister(left),
                           g.UseRegister(right), cont);
            }
          }
          break;
        case kSignedLessThan:
        case kSignedGreaterThanOrEqual:
        case kUnsignedLessThan:
        case kUnsignedGreaterThanOrEqual: {
          Int32BinopMatcher m(node, true);
          if (m.right().Is(0)) {
            VisitWordCompareZero(selector, g.UseRegisterOrImmediateZero(left),
                                 cont);
          } else {
            VisitCompare(selector, opcode, g.UseRegister(left),
                         g.UseImmediate(right), cont);
          }
        } break;
        default:
          Int32BinopMatcher m(node, true);
          if (m.right().Is(0)) {
            VisitWordCompareZero(selector, g.UseRegisterOrImmediateZero(left),
                                 cont);
          } else {
            VisitCompare(selector, opcode, g.UseRegister(left),
                         g.UseRegister(right), cont);
          }
      }
    }
  } else {
    VisitCompare(selector, opcode, g.UseRegister(left), g.UseRegister(right),
                 cont);
  }
}

void VisitWordCompare(InstructionSelector* selector, Node* node,
                      FlagsContinuation* cont) {
  VisitWordCompare(selector, node, kRiscvCmp, cont, false);
}

void EmitWordCompareZero(InstructionSelector* selector, Node* value,
                         FlagsContinuation* cont) {
  RiscvOperandGenerator g(selector);
  selector->EmitWithContinuation(kRiscvCmpZero,
                                 g.UseRegisterOrImmediateZero(value), cont);
}

void VisitAtomicLoad(InstructionSelector* selector, Node* node,
                     ArchOpcode opcode, AtomicWidth width) {
  RiscvOperandGenerator g(selector);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);
  if (g.CanBeImmediate(index, opcode)) {
    selector->Emit(opcode | AddressingModeField::encode(kMode_MRI) |
                       AtomicWidthField::encode(width),
                   g.DefineAsRegister(node), g.UseRegister(base),
                   g.UseImmediate(index));
  } else {
    InstructionOperand addr_reg = g.TempRegister();
    selector->Emit(kRiscvAdd | AddressingModeField::encode(kMode_None),
                   addr_reg, g.UseRegister(index), g.UseRegister(base));
    // Emit desired load opcode, using temp addr_reg.
    selector->Emit(opcode | AddressingModeField::encode(kMode_MRI) |
                       AtomicWidthField::encode(width),
                   g.DefineAsRegister(node), addr_reg, g.TempImmediate(0));
  }
}

void VisitAtomicStore(InstructionSelector* selector, Node* node,
                      ArchOpcode opcode, AtomicWidth width) {
  RiscvOperandGenerator g(selector);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);
  Node* value = node->InputAt(2);

  if (g.CanBeImmediate(index, opcode)) {
    selector->Emit(opcode | AddressingModeField::encode(kMode_MRI) |
                       AtomicWidthField::encode(width),
                   g.NoOutput(), g.UseRegister(base), g.UseImmediate(index),
                   g.UseRegisterOrImmediateZero(value));
  } else {
    InstructionOperand addr_reg = g.TempRegister();
    selector->Emit(kRiscvAdd | AddressingModeField::encode(kMode_None),
                   addr_reg, g.UseRegister(index), g.UseRegister(base));
    // Emit desired store opcode, using temp addr_reg.
    selector->Emit(opcode | AddressingModeField::encode(kMode_MRI) |
                       AtomicWidthField::encode(width),
                   g.NoOutput(), addr_reg, g.TempImmediate(0),
                   g.UseRegisterOrImmediateZero(value));
  }
}

void VisitAtomicExchange(InstructionSelector* selector, Node* node,
                         ArchOpcode opcode, AtomicWidth width) {
  RiscvOperandGenerator g(selector);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);
  Node* value = node->InputAt(2);

  AddressingMode addressing_mode = kMode_MRI;
  InstructionOperand inputs[3];
  size_t input_count = 0;
  inputs[input_count++] = g.UseUniqueRegister(base);
  inputs[input_count++] = g.UseUniqueRegister(index);
  inputs[input_count++] = g.UseUniqueRegister(value);
  InstructionOperand outputs[1];
  outputs[0] = g.UseUniqueRegister(node);
  InstructionOperand temp[3];
  temp[0] = g.TempRegister();
  temp[1] = g.TempRegister();
  temp[2] = g.TempRegister();
  InstructionCode code = opcode | AddressingModeField::encode(addressing_mode) |
                         AtomicWidthField::encode(width);
  selector->Emit(code, 1, outputs, input_count, inputs, 3, temp);
}

void VisitAtomicCompareExchange(InstructionSelector* selector, Node* node,
                                ArchOpcode opcode, AtomicWidth width) {
  RiscvOperandGenerator g(selector);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);
  Node* old_value = node->InputAt(2);
  Node* new_value = node->InputAt(3);

  AddressingMode addressing_mode = kMode_MRI;
  InstructionOperand inputs[4];
  size_t input_count = 0;
  inputs[input_count++] = g.UseUniqueRegister(base);
  inputs[input_count++] = g.UseUniqueRegister(index);
  inputs[input_count++] = g.UseUniqueRegister(old_value);
  inputs[input_count++] = g.UseUniqueRegister(new_value);
  InstructionOperand outputs[1];
  outputs[0] = g.UseUniqueRegister(node);
  InstructionOperand temp[3];
  temp[0] = g.TempRegister();
  temp[1] = g.TempRegister();
  temp[2] = g.TempRegister();
  InstructionCode code = opcode | AddressingModeField::encode(addressing_mode) |
                         AtomicWidthField::encode(width);
  selector->Emit(code, 1, outputs, input_count, inputs, 3, temp);
}

void VisitAtomicBinop(InstructionSelector* selector, Node* node,
                      ArchOpcode opcode) {
  RiscvOperandGenerator g(selector);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);
  Node* value = node->InputAt(2);

  AddressingMode addressing_mode = kMode_MRI;
  InstructionOperand inputs[3];
  size_t input_count = 0;
  inputs[input_count++] = g.UseUniqueRegister(base);
  inputs[input_count++] = g.UseUniqueRegister(index);
  inputs[input_count++] = g.UseUniqueRegister(value);
  InstructionOperand outputs[1];
  outputs[0] = g.UseUniqueRegister(node);
  InstructionOperand temps[4];
  temps[0] = g.TempRegister();
  temps[1] = g.TempRegister();
  temps[2] = g.TempRegister();
  temps[3] = g.TempRegister();
  InstructionCode code = opcode | AddressingModeField::encode(addressing_mode);
  selector->Emit(code, 1, outputs, input_count, inputs, 4, temps);
}

}  // namespace

void InstructionSelector::VisitStackPointerGreaterThan(
    Node* node, FlagsContinuation* cont) {
  StackCheckKind kind = StackCheckKindOf(node->op());
  InstructionCode opcode =
      kArchStackPointerGreaterThan | MiscField::encode(static_cast<int>(kind));

  RiscvOperandGenerator g(this);

  // No outputs.
  InstructionOperand* const outputs = nullptr;
  const int output_count = 0;

  // Applying an offset to this stack check requires a temp register. Offsets
  // are only applied to the first stack check. If applying an offset, we must
  // ensure the input and temp registers do not alias, thus kUniqueRegister.
  InstructionOperand temps[] = {g.TempRegister()};
  const int temp_count = (kind == StackCheckKind::kJSFunctionEntry ? 1 : 0);
  const auto register_mode = (kind == StackCheckKind::kJSFunctionEntry)
                                 ? OperandGenerator::kUniqueRegister
                                 : OperandGenerator::kRegister;

  Node* const value = node->InputAt(0);
  InstructionOperand inputs[] = {g.UseRegisterWithMode(value, register_mode)};
  static constexpr int input_count = arraysize(inputs);

  EmitWithContinuation(opcode, output_count, outputs, input_count, inputs,
                       temp_count, temps, cont);
}

// Shared routine for word comparisons against zero.
void InstructionSelector::VisitWordCompareZero(Node* user, Node* value,
                                               FlagsContinuation* cont) {
  // Try to combine with comparisons against 0 by simply inverting the branch.
  while (CanCover(user, value)) {
    if (value->opcode() == IrOpcode::kWord32Equal) {
      Int32BinopMatcher m(value);
      if (!m.right().Is(0)) break;
      user = value;
      value = m.left().node();
    } else if (value->opcode() == IrOpcode::kWord64Equal) {
      Int64BinopMatcher m(value);
      if (!m.right().Is(0)) break;
      user = value;
      value = m.left().node();
    } else {
      break;
    }

    cont->Negate();
  }

  if (CanCover(user, value)) {
    switch (value->opcode()) {
      case IrOpcode::kWord32Equal:
        cont->OverwriteAndNegateIfEqual(kEqual);
        return VisitWordCompare(this, value, cont);
      case IrOpcode::kInt32LessThan:
        cont->OverwriteAndNegateIfEqual(kSignedLessThan);
        return VisitWordCompare(this, value, cont);
      case IrOpcode::kInt32LessThanOrEqual:
        cont->OverwriteAndNegateIfEqual(kSignedLessThanOrEqual);
        return VisitWordCompare(this, value, cont);
      case IrOpcode::kUint32LessThan:
        cont->OverwriteAndNegateIfEqual(kUnsignedLessThan);
        return VisitWordCompare(this, value, cont);
      case IrOpcode::kUint32LessThanOrEqual:
        cont->OverwriteAndNegateIfEqual(kUnsignedLessThanOrEqual);
        return VisitWordCompare(this, value, cont);
      case IrOpcode::kFloat32Equal:
        cont->OverwriteAndNegateIfEqual(kEqual);
        return VisitFloat32Compare(this, value, cont);
      case IrOpcode::kFloat32LessThan:
        cont->OverwriteAndNegateIfEqual(kUnsignedLessThan);
        return VisitFloat32Compare(this, value, cont);
      case IrOpcode::kFloat32LessThanOrEqual:
        cont->OverwriteAndNegateIfEqual(kUnsignedLessThanOrEqual);
        return VisitFloat32Compare(this, value, cont);
      case IrOpcode::kFloat64Equal:
        cont->OverwriteAndNegateIfEqual(kEqual);
        return VisitFloat64Compare(this, value, cont);
      case IrOpcode::kFloat64LessThan:
        cont->OverwriteAndNegateIfEqual(kUnsignedLessThan);
        return VisitFloat64Compare(this, value, cont);
      case IrOpcode::kFloat64LessThanOrEqual:
        cont->OverwriteAndNegateIfEqual(kUnsignedLessThanOrEqual);
        return VisitFloat64Compare(this, value, cont);
      case IrOpcode::kProjection:
        // Check if this is the overflow output projection of an
        // <Operation>WithOverflow node.
        if (ProjectionIndexOf(value->op()) == 1u) {
          // We cannot combine the <Operation>WithOverflow with this branch
          // unless the 0th projection (the use of the actual value of the
          // <Operation> is either nullptr, which means there's no use of the
          // actual value, or was already defined, which means it is scheduled
          // *AFTER* this branch).
          Node* const node = value->InputAt(0);
          Node* const result = NodeProperties::FindProjection(node, 0);
          if (result == nullptr || IsDefined(result)) {
            switch (node->opcode()) {
              case IrOpcode::kInt32AddWithOverflow:
                cont->OverwriteAndNegateIfEqual(kOverflow);
                return VisitBinop(this, node, kRiscvAddOvf, cont);
              case IrOpcode::kInt32SubWithOverflow:
                cont->OverwriteAndNegateIfEqual(kOverflow);
                return VisitBinop(this, node, kRiscvSubOvf, cont);
              case IrOpcode::kInt32MulWithOverflow:
                cont->OverwriteAndNegateIfEqual(kOverflow);
                return VisitBinop(this, node, kRiscvMulOvf32, cont);
              case IrOpcode::kInt64AddWithOverflow:
              case IrOpcode::kInt64SubWithOverflow:
                TRACE_UNIMPL();
                break;
              default:
                break;
            }
          }
        }
        break;
      case IrOpcode::kWord32And:
        return VisitWordCompare(this, value, kRiscvTst, cont, true);
      case IrOpcode::kStackPointerGreaterThan:
        cont->OverwriteAndNegateIfEqual(kStackPointerGreaterThanCondition);
        return VisitStackPointerGreaterThan(value, cont);
      default:
        break;
    }
  }

  // Continuation could not be combined with a compare, emit compare against 0.
  EmitWordCompareZero(this, value, cont);
}

void InstructionSelector::VisitSwitch(Node* node, const SwitchInfo& sw) {
  RiscvOperandGenerator g(this);
  InstructionOperand value_operand = g.UseRegister(node->InputAt(0));

  // Emit either ArchTableSwitch or ArchBinarySearchSwitch.
  if (enable_switch_jump_table_ == kEnableSwitchJumpTable) {
    static const size_t kMaxTableSwitchValueRange = 2 << 16;
    size_t table_space_cost = 10 + 2 * sw.value_range();
    size_t table_time_cost = 3;
    size_t lookup_space_cost = 2 + 2 * sw.case_count();
    size_t lookup_time_cost = sw.case_count();
    if (sw.case_count() > 0 &&
        table_space_cost + 3 * table_time_cost <=
            lookup_space_cost + 3 * lookup_time_cost &&
        sw.min_value() > std::numeric_limits<int32_t>::min() &&
        sw.value_range() <= kMaxTableSwitchValueRange) {
      InstructionOperand index_operand = value_operand;
      if (sw.min_value()) {
        index_operand = g.TempRegister();
        Emit(kRiscvSub, index_operand, value_operand,
             g.TempImmediate(sw.min_value()));
      }
      // Generate a table lookup.
      return EmitTableSwitch(sw, index_operand);
    }
  }

  // Generate a tree of conditional jumps.
  return EmitBinarySearchSwitch(sw, value_operand);
}

void InstructionSelector::VisitWord32Equal(Node* const node) {
  FlagsContinuation cont = FlagsContinuation::ForSet(kEqual, node);
  Int32BinopMatcher m(node);
  if (m.right().Is(0)) {
    return VisitWordCompareZero(m.node(), m.left().node(), &cont);
  }

  VisitWordCompare(this, node, &cont);
}

void InstructionSelector::VisitInt32LessThan(Node* node) {
  FlagsContinuation cont = FlagsContinuation::ForSet(kSignedLessThan, node);
  VisitWordCompare(this, node, &cont);
}

void InstructionSelector::VisitInt32LessThanOrEqual(Node* node) {
  FlagsContinuation cont =
      FlagsContinuation::ForSet(kSignedLessThanOrEqual, node);
  VisitWordCompare(this, node, &cont);
}

void InstructionSelector::VisitUint32LessThan(Node* node) {
  FlagsContinuation cont = FlagsContinuation::ForSet(kUnsignedLessThan, node);
  VisitWordCompare(this, node, &cont);
}

void InstructionSelector::VisitUint32LessThanOrEqual(Node* node) {
  FlagsContinuation cont =
      FlagsContinuation::ForSet(kUnsignedLessThanOrEqual, node);
  VisitWordCompare(this, node, &cont);
}

void InstructionSelector::VisitInt32AddWithOverflow(Node* node) {
  if (Node* ovf = NodeProperties::FindProjection(node, 1)) {
    FlagsContinuation cont = FlagsContinuation::ForSet(kOverflow, ovf);
    return VisitBinop(this, node, kRiscvAddOvf, &cont);
  }
  FlagsContinuation cont;
  VisitBinop(this, node, kRiscvAddOvf, &cont);
}

void InstructionSelector::VisitInt32SubWithOverflow(Node* node) {
  if (Node* ovf = NodeProperties::FindProjection(node, 1)) {
    FlagsContinuation cont = FlagsContinuation::ForSet(kOverflow, ovf);
    return VisitBinop(this, node, kRiscvSubOvf, &cont);
  }
  FlagsContinuation cont;
  VisitBinop(this, node, kRiscvSubOvf, &cont);
}

void InstructionSelector::VisitInt32MulWithOverflow(Node* node) {
  if (Node* ovf = NodeProperties::FindProjection(node, 1)) {
    FlagsContinuation cont = FlagsContinuation::ForSet(kOverflow, ovf);
    return VisitBinop(this, node, kRiscvMulOvf32, &cont);
  }
  FlagsContinuation cont;
  VisitBinop(this, node, kRiscvMulOvf32, &cont);
}

void InstructionSelector::VisitFloat32Equal(Node* node) {
  FlagsContinuation cont = FlagsContinuation::ForSet(kEqual, node);
  VisitFloat32Compare(this, node, &cont);
}

void InstructionSelector::VisitFloat32LessThan(Node* node) {
  FlagsContinuation cont = FlagsContinuation::ForSet(kUnsignedLessThan, node);
  VisitFloat32Compare(this, node, &cont);
}

void InstructionSelector::VisitFloat32LessThanOrEqual(Node* node) {
  FlagsContinuation cont =
      FlagsContinuation::ForSet(kUnsignedLessThanOrEqual, node);
  VisitFloat32Compare(this, node, &cont);
}

void InstructionSelector::VisitFloat64Equal(Node* node) {
  FlagsContinuation cont = FlagsContinuation::ForSet(kEqual, node);
  VisitFloat64Compare(this, node, &cont);
}

void InstructionSelector::VisitFloat64LessThan(Node* node) {
  FlagsContinuation cont = FlagsContinuation::ForSet(kUnsignedLessThan, node);
  VisitFloat64Compare(this, node, &cont);
}

void InstructionSelector::VisitFloat64LessThanOrEqual(Node* node) {
  FlagsContinuation cont =
      FlagsContinuation::ForSet(kUnsignedLessThanOrEqual, node);
  VisitFloat64Compare(this, node, &cont);
}

void InstructionSelector::VisitFloat64ExtractLowWord32(Node* node) {
  VisitRR(this, kRiscvFloat64ExtractLowWord32, node);
}

void InstructionSelector::VisitFloat64ExtractHighWord32(Node* node) {
  VisitRR(this, kRiscvFloat64ExtractHighWord32, node);
}

void InstructionSelector::VisitFloat64SilenceNaN(Node* node) {
  VisitRR(this, kRiscvFloat64SilenceNaN, node);
}

void InstructionSelector::VisitFloat64InsertLowWord32(Node* node) {
  RiscvOperandGenerator g(this);
  Node* left = node->InputAt(0);
  Node* right = node->InputAt(1);
  Emit(kRiscvFloat64InsertLowWord32, g.DefineSameAsFirst(node),
       g.UseRegister(left), g.UseRegister(right));
}

void InstructionSelector::VisitFloat64InsertHighWord32(Node* node) {
  RiscvOperandGenerator g(this);
  Node* left = node->InputAt(0);
  Node* right = node->InputAt(1);
  Emit(kRiscvFloat64InsertHighWord32, g.DefineSameAsFirst(node),
       g.UseRegister(left), g.UseRegister(right));
}

void InstructionSelector::VisitMemoryBarrier(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvSync, g.NoOutput());
}

void InstructionSelector::VisitWord32AtomicLoad(Node* node) {
  AtomicLoadParameters atomic_load_params = AtomicLoadParametersOf(node->op());
  LoadRepresentation load_rep = atomic_load_params.representation();
  ArchOpcode opcode;
  switch (load_rep.representation()) {
    case MachineRepresentation::kWord8:
      opcode = load_rep.IsSigned() ? kAtomicLoadInt8 : kAtomicLoadUint8;
      break;
    case MachineRepresentation::kWord16:
      opcode = load_rep.IsSigned() ? kAtomicLoadInt16 : kAtomicLoadUint16;
      break;
    case MachineRepresentation::kTaggedSigned:   // Fall through.
    case MachineRepresentation::kTaggedPointer:  // Fall through.
    case MachineRepresentation::kTagged:
    case MachineRepresentation::kWord32:
      opcode = kAtomicLoadWord32;
      break;
    default:
      UNREACHABLE();
  }
  VisitAtomicLoad(this, node, opcode, AtomicWidth::kWord32);
}

void InstructionSelector::VisitWord32AtomicStore(Node* node) {
  AtomicStoreParameters store_params = AtomicStoreParametersOf(node->op());
  MachineRepresentation rep = store_params.representation();
  ArchOpcode opcode;
  switch (rep) {
    case MachineRepresentation::kWord8:
      opcode = kAtomicStoreWord8;
      break;
    case MachineRepresentation::kWord16:
      opcode = kAtomicStoreWord16;
      break;
    case MachineRepresentation::kTaggedSigned:   // Fall through.
    case MachineRepresentation::kTaggedPointer:  // Fall through.
    case MachineRepresentation::kTagged:
    case MachineRepresentation::kWord32:
      opcode = kAtomicStoreWord32;
      break;
    default:
      UNREACHABLE();
  }

  VisitAtomicStore(this, node, opcode, AtomicWidth::kWord32);
}

void InstructionSelector::VisitWord32AtomicExchange(Node* node) {
  ArchOpcode opcode;
  MachineType type = AtomicOpType(node->op());
  if (type == MachineType::Int8()) {
    opcode = kAtomicExchangeInt8;
  } else if (type == MachineType::Uint8()) {
    opcode = kAtomicExchangeUint8;
  } else if (type == MachineType::Int16()) {
    opcode = kAtomicExchangeInt16;
  } else if (type == MachineType::Uint16()) {
    opcode = kAtomicExchangeUint16;
  } else if (type == MachineType::Int32() || type == MachineType::Uint32()) {
    opcode = kAtomicExchangeWord32;
  } else {
    UNREACHABLE();
  }

  VisitAtomicExchange(this, node, opcode, AtomicWidth::kWord32);
}

void InstructionSelector::VisitWord32AtomicCompareExchange(Node* node) {
  ArchOpcode opcode;
  MachineType type = AtomicOpType(node->op());
  if (type == MachineType::Int8()) {
    opcode = kAtomicCompareExchangeInt8;
  } else if (type == MachineType::Uint8()) {
    opcode = kAtomicCompareExchangeUint8;
  } else if (type == MachineType::Int16()) {
    opcode = kAtomicCompareExchangeInt16;
  } else if (type == MachineType::Uint16()) {
    opcode = kAtomicCompareExchangeUint16;
  } else if (type == MachineType::Int32() || type == MachineType::Uint32()) {
    opcode = kAtomicCompareExchangeWord32;
  } else {
    UNREACHABLE();
  }

  VisitAtomicCompareExchange(this, node, opcode, AtomicWidth::kWord32);
}

void InstructionSelector::VisitWord32AtomicBinaryOperation(
    Node* node, ArchOpcode int8_op, ArchOpcode uint8_op, ArchOpcode int16_op,
    ArchOpcode uint16_op, ArchOpcode word32_op) {
  ArchOpcode opcode;
  MachineType type = AtomicOpType(node->op());
  if (type == MachineType::Int8()) {
    opcode = int8_op;
  } else if (type == MachineType::Uint8()) {
    opcode = uint8_op;
  } else if (type == MachineType::Int16()) {
    opcode = int16_op;
  } else if (type == MachineType::Uint16()) {
    opcode = uint16_op;
  } else if (type == MachineType::Int32() || type == MachineType::Uint32()) {
    opcode = word32_op;
  } else {
    UNREACHABLE();
  }

  VisitAtomicBinop(this, node, opcode);
}

#define VISIT_ATOMIC_BINOP(op)                                           \
  void InstructionSelector::VisitWord32Atomic##op(Node* node) {          \
    VisitWord32AtomicBinaryOperation(                                    \
        node, kAtomic##op##Int8, kAtomic##op##Uint8, kAtomic##op##Int16, \
        kAtomic##op##Uint16, kAtomic##op##Word32);                       \
  }
VISIT_ATOMIC_BINOP(Add)
VISIT_ATOMIC_BINOP(Sub)
VISIT_ATOMIC_BINOP(And)
VISIT_ATOMIC_BINOP(Or)
VISIT_ATOMIC_BINOP(Xor)
#undef VISIT_ATOMIC_BINOP

void InstructionSelector::VisitInt32AbsWithOverflow(Node* node) {
  UNREACHABLE();
}

void InstructionSelector::VisitInt64AbsWithOverflow(Node* node) {
  UNREACHABLE();
}

#define SIMD_TYPE_LIST(V) \
  V(F32x4)                \
  V(I64x2)                \
  V(I32x4)                \
  V(I16x8)                \
  V(I8x16)

#define SIMD_UNOP_LIST(V)                                       \
  V(F64x2Abs, kRiscvF64x2Abs)                                   \
  V(F64x2Neg, kRiscvF64x2Neg)                                   \
  V(F64x2Sqrt, kRiscvF64x2Sqrt)                                 \
  V(F64x2ConvertLowI32x4S, kRiscvF64x2ConvertLowI32x4S)         \
  V(F64x2ConvertLowI32x4U, kRiscvF64x2ConvertLowI32x4U)         \
  V(F64x2PromoteLowF32x4, kRiscvF64x2PromoteLowF32x4)           \
  V(F64x2Ceil, kRiscvF64x2Ceil)                                 \
  V(F64x2Floor, kRiscvF64x2Floor)                               \
  V(F64x2Trunc, kRiscvF64x2Trunc)                               \
  V(F64x2NearestInt, kRiscvF64x2NearestInt)                     \
  V(I64x2Neg, kRiscvI64x2Neg)                                   \
  V(I64x2Abs, kRiscvI64x2Abs)                                   \
  V(I64x2BitMask, kRiscvI64x2BitMask)                           \
  V(F32x4SConvertI32x4, kRiscvF32x4SConvertI32x4)               \
  V(F32x4UConvertI32x4, kRiscvF32x4UConvertI32x4)               \
  V(F32x4Abs, kRiscvF32x4Abs)                                   \
  V(F32x4Neg, kRiscvF32x4Neg)                                   \
  V(F32x4Sqrt, kRiscvF32x4Sqrt)                                 \
  V(F32x4RecipApprox, kRiscvF32x4RecipApprox)                   \
  V(F32x4RecipSqrtApprox, kRiscvF32x4RecipSqrtApprox)           \
  V(F32x4DemoteF64x2Zero, kRiscvF32x4DemoteF64x2Zero)           \
  V(F32x4Ceil, kRiscvF32x4Ceil)                                 \
  V(F32x4Floor, kRiscvF32x4Floor)                               \
  V(F32x4Trunc, kRiscvF32x4Trunc)                               \
  V(F32x4NearestInt, kRiscvF32x4NearestInt)                     \
  V(I32x4RelaxedTruncF32x4S, kRiscvI32x4SConvertF32x4)          \
  V(I32x4RelaxedTruncF32x4U, kRiscvI32x4UConvertF32x4)          \
  V(I32x4RelaxedTruncF64x2SZero, kRiscvI32x4TruncSatF64x2SZero) \
  V(I32x4RelaxedTruncF64x2UZero, kRiscvI32x4TruncSatF64x2UZero) \
  V(I64x2SConvertI32x4Low, kRiscvI64x2SConvertI32x4Low)         \
  V(I64x2SConvertI32x4High, kRiscvI64x2SConvertI32x4High)       \
  V(I64x2UConvertI32x4Low, kRiscvI64x2UConvertI32x4Low)         \
  V(I64x2UConvertI32x4High, kRiscvI64x2UConvertI32x4High)       \
  V(I32x4SConvertF32x4, kRiscvI32x4SConvertF32x4)               \
  V(I32x4UConvertF32x4, kRiscvI32x4UConvertF32x4)               \
  V(I32x4Neg, kRiscvI32x4Neg)                                   \
  V(I32x4SConvertI16x8Low, kRiscvI32x4SConvertI16x8Low)         \
  V(I32x4SConvertI16x8High, kRiscvI32x4SConvertI16x8High)       \
  V(I32x4UConvertI16x8Low, kRiscvI32x4UConvertI16x8Low)         \
  V(I32x4UConvertI16x8High, kRiscvI32x4UConvertI16x8High)       \
  V(I32x4Abs, kRiscvI32x4Abs)                                   \
  V(I32x4BitMask, kRiscvI32x4BitMask)                           \
  V(I32x4TruncSatF64x2SZero, kRiscvI32x4TruncSatF64x2SZero)     \
  V(I32x4TruncSatF64x2UZero, kRiscvI32x4TruncSatF64x2UZero)     \
  V(I16x8Neg, kRiscvI16x8Neg)                                   \
  V(I16x8SConvertI8x16Low, kRiscvI16x8SConvertI8x16Low)         \
  V(I16x8SConvertI8x16High, kRiscvI16x8SConvertI8x16High)       \
  V(I16x8UConvertI8x16Low, kRiscvI16x8UConvertI8x16Low)         \
  V(I16x8UConvertI8x16High, kRiscvI16x8UConvertI8x16High)       \
  V(I16x8Abs, kRiscvI16x8Abs)                                   \
  V(I16x8BitMask, kRiscvI16x8BitMask)                           \
  V(I8x16Neg, kRiscvI8x16Neg)                                   \
  V(I8x16Abs, kRiscvI8x16Abs)                                   \
  V(I8x16BitMask, kRiscvI8x16BitMask)                           \
  V(I8x16Popcnt, kRiscvI8x16Popcnt)                             \
  V(S128Not, kRiscvS128Not)                                     \
  V(V128AnyTrue, kRiscvV128AnyTrue)                             \
  V(I32x4AllTrue, kRiscvI32x4AllTrue)                           \
  V(I16x8AllTrue, kRiscvI16x8AllTrue)                           \
  V(I8x16AllTrue, kRiscvI8x16AllTrue)                           \
  V(I64x2AllTrue, kRiscvI64x2AllTrue)

#define SIMD_SHIFT_OP_LIST(V) \
  V(I64x2Shl)                 \
  V(I64x2ShrS)                \
  V(I64x2ShrU)                \
  V(I32x4Shl)                 \
  V(I32x4ShrS)                \
  V(I32x4ShrU)                \
  V(I16x8Shl)                 \
  V(I16x8ShrS)                \
  V(I16x8ShrU)                \
  V(I8x16Shl)                 \
  V(I8x16ShrS)                \
  V(I8x16ShrU)

#define SIMD_BINOP_LIST(V)                              \
  V(F64x2Add, kRiscvF64x2Add)                           \
  V(F64x2Sub, kRiscvF64x2Sub)                           \
  V(F64x2Mul, kRiscvF64x2Mul)                           \
  V(F64x2Div, kRiscvF64x2Div)                           \
  V(F64x2Min, kRiscvF64x2Min)                           \
  V(F64x2Max, kRiscvF64x2Max)                           \
  V(F64x2Eq, kRiscvF64x2Eq)                             \
  V(F64x2Ne, kRiscvF64x2Ne)                             \
  V(F64x2Lt, kRiscvF64x2Lt)                             \
  V(F64x2Le, kRiscvF64x2Le)                             \
  V(I64x2Eq, kRiscvI64x2Eq)                             \
  V(I64x2Ne, kRiscvI64x2Ne)                             \
  V(I64x2GtS, kRiscvI64x2GtS)                           \
  V(I64x2GeS, kRiscvI64x2GeS)                           \
  V(I64x2Add, kRiscvI64x2Add)                           \
  V(I64x2Sub, kRiscvI64x2Sub)                           \
  V(I64x2Mul, kRiscvI64x2Mul)                           \
  V(F32x4Add, kRiscvF32x4Add)                           \
  V(F32x4Sub, kRiscvF32x4Sub)                           \
  V(F32x4Mul, kRiscvF32x4Mul)                           \
  V(F32x4Div, kRiscvF32x4Div)                           \
  V(F32x4Max, kRiscvF32x4Max)                           \
  V(F32x4Min, kRiscvF32x4Min)                           \
  V(F32x4Eq, kRiscvF32x4Eq)                             \
  V(F32x4Ne, kRiscvF32x4Ne)                             \
  V(F32x4Lt, kRiscvF32x4Lt)                             \
  V(F32x4Le, kRiscvF32x4Le)                             \
  V(F32x4RelaxedMin, kRiscvF32x4Min)                    \
  V(F32x4RelaxedMax, kRiscvF32x4Max)                    \
  V(F64x2RelaxedMin, kRiscvF64x2Min)                    \
  V(F64x2RelaxedMax, kRiscvF64x2Max)                    \
  V(I32x4Add, kRiscvI32x4Add)                           \
  V(I32x4Sub, kRiscvI32x4Sub)                           \
  V(I32x4Mul, kRiscvI32x4Mul)                           \
  V(I32x4MaxS, kRiscvI32x4MaxS)                         \
  V(I32x4MinS, kRiscvI32x4MinS)                         \
  V(I32x4MaxU, kRiscvI32x4MaxU)                         \
  V(I32x4MinU, kRiscvI32x4MinU)                         \
  V(I32x4Eq, kRiscvI32x4Eq)                             \
  V(I32x4Ne, kRiscvI32x4Ne)                             \
  V(I32x4GtS, kRiscvI32x4GtS)                           \
  V(I32x4GeS, kRiscvI32x4GeS)                           \
  V(I32x4GtU, kRiscvI32x4GtU)                           \
  V(I32x4GeU, kRiscvI32x4GeU)                           \
  V(I16x8Add, kRiscvI16x8Add)                           \
  V(I16x8AddSatS, kRiscvI16x8AddSatS)                   \
  V(I16x8AddSatU, kRiscvI16x8AddSatU)                   \
  V(I16x8Sub, kRiscvI16x8Sub)                           \
  V(I16x8SubSatS, kRiscvI16x8SubSatS)                   \
  V(I16x8SubSatU, kRiscvI16x8SubSatU)                   \
  V(I16x8Mul, kRiscvI16x8Mul)                           \
  V(I16x8MaxS, kRiscvI16x8MaxS)                         \
  V(I16x8MinS, kRiscvI16x8MinS)                         \
  V(I16x8MaxU, kRiscvI16x8MaxU)                         \
  V(I16x8MinU, kRiscvI16x8MinU)                         \
  V(I16x8Eq, kRiscvI16x8Eq)                             \
  V(I16x8Ne, kRiscvI16x8Ne)                             \
  V(I16x8GtS, kRiscvI16x8GtS)                           \
  V(I16x8GeS, kRiscvI16x8GeS)                           \
  V(I16x8GtU, kRiscvI16x8GtU)                           \
  V(I16x8GeU, kRiscvI16x8GeU)                           \
  V(I16x8RoundingAverageU, kRiscvI16x8RoundingAverageU) \
  V(I16x8Q15MulRSatS, kRiscvI16x8Q15MulRSatS)           \
  V(I16x8SConvertI32x4, kRiscvI16x8SConvertI32x4)       \
  V(I16x8UConvertI32x4, kRiscvI16x8UConvertI32x4)       \
  V(I8x16Add, kRiscvI8x16Add)                           \
  V(I8x16AddSatS, kRiscvI8x16AddSatS)                   \
  V(I8x16AddSatU, kRiscvI8x16AddSatU)                   \
  V(I8x16Sub, kRiscvI8x16Sub)                           \
  V(I8x16SubSatS, kRiscvI8x16SubSatS)                   \
  V(I8x16SubSatU, kRiscvI8x16SubSatU)                   \
  V(I8x16MaxS, kRiscvI8x16MaxS)                         \
  V(I8x16MinS, kRiscvI8x16MinS)                         \
  V(I8x16MaxU, kRiscvI8x16MaxU)                         \
  V(I8x16MinU, kRiscvI8x16MinU)                         \
  V(I8x16Eq, kRiscvI8x16Eq)                             \
  V(I8x16Ne, kRiscvI8x16Ne)                             \
  V(I8x16GtS, kRiscvI8x16GtS)                           \
  V(I8x16GeS, kRiscvI8x16GeS)                           \
  V(I8x16GtU, kRiscvI8x16GtU)                           \
  V(I8x16GeU, kRiscvI8x16GeU)                           \
  V(I8x16RoundingAverageU, kRiscvI8x16RoundingAverageU) \
  V(I8x16SConvertI16x8, kRiscvI8x16SConvertI16x8)       \
  V(I8x16UConvertI16x8, kRiscvI8x16UConvertI16x8)       \
  V(S128And, kRiscvS128And)                             \
  V(S128Or, kRiscvS128Or)                               \
  V(S128Xor, kRiscvS128Xor)                             \
  V(S128AndNot, kRiscvS128AndNot)

void InstructionSelector::VisitS128Const(Node* node) {
  RiscvOperandGenerator g(this);
  static const int kUint32Immediates = kSimd128Size / sizeof(uint32_t);
  uint32_t val[kUint32Immediates];
  memcpy(val, S128ImmediateParameterOf(node->op()).data(), kSimd128Size);
  // If all bytes are zeros or ones, avoid emitting code for generic constants
  bool all_zeros = !(val[0] || val[1] || val[2] || val[3]);
  bool all_ones = val[0] == UINT32_MAX && val[1] == UINT32_MAX &&
                  val[2] == UINT32_MAX && val[3] == UINT32_MAX;
  InstructionOperand dst = g.DefineAsRegister(node);
  if (all_zeros) {
    Emit(kRiscvS128Zero, dst);
  } else if (all_ones) {
    Emit(kRiscvS128AllOnes, dst);
  } else {
    Emit(kRiscvS128Const, dst, g.UseImmediate(val[0]), g.UseImmediate(val[1]),
         g.UseImmediate(val[2]), g.UseImmediate(val[3]));
  }
}

void InstructionSelector::VisitS128Zero(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvS128Zero, g.DefineAsRegister(node));
}

#define SIMD_VISIT_SPLAT(Type)                               \
  void InstructionSelector::Visit##Type##Splat(Node* node) { \
    VisitRR(this, kRiscv##Type##Splat, node);                \
  }
SIMD_TYPE_LIST(SIMD_VISIT_SPLAT)
SIMD_VISIT_SPLAT(F64x2)
#undef SIMD_VISIT_SPLAT

#define SIMD_VISIT_EXTRACT_LANE(Type, Sign)                              \
  void InstructionSelector::Visit##Type##ExtractLane##Sign(Node* node) { \
    VisitRRI(this, kRiscv##Type##ExtractLane##Sign, node);               \
  }
SIMD_VISIT_EXTRACT_LANE(F64x2, )
SIMD_VISIT_EXTRACT_LANE(F32x4, )
SIMD_VISIT_EXTRACT_LANE(I32x4, )
SIMD_VISIT_EXTRACT_LANE(I64x2, )
SIMD_VISIT_EXTRACT_LANE(I16x8, U)
SIMD_VISIT_EXTRACT_LANE(I16x8, S)
SIMD_VISIT_EXTRACT_LANE(I8x16, U)
SIMD_VISIT_EXTRACT_LANE(I8x16, S)
#undef SIMD_VISIT_EXTRACT_LANE

#define SIMD_VISIT_REPLACE_LANE(Type)                              \
  void InstructionSelector::Visit##Type##ReplaceLane(Node* node) { \
    VisitRRIR(this, kRiscv##Type##ReplaceLane, node);              \
  }
SIMD_TYPE_LIST(SIMD_VISIT_REPLACE_LANE)
SIMD_VISIT_REPLACE_LANE(F64x2)
#undef SIMD_VISIT_REPLACE_LANE

#define SIMD_VISIT_UNOP(Name, instruction)            \
  void InstructionSelector::Visit##Name(Node* node) { \
    VisitRR(this, instruction, node);                 \
  }
SIMD_UNOP_LIST(SIMD_VISIT_UNOP)
#undef SIMD_VISIT_UNOP

#define SIMD_VISIT_SHIFT_OP(Name)                     \
  void InstructionSelector::Visit##Name(Node* node) { \
    VisitSimdShift(this, kRiscv##Name, node);         \
  }
SIMD_SHIFT_OP_LIST(SIMD_VISIT_SHIFT_OP)
#undef SIMD_VISIT_SHIFT_OP

#define SIMD_VISIT_BINOP(Name, instruction)           \
  void InstructionSelector::Visit##Name(Node* node) { \
    VisitRRR(this, instruction, node);                \
  }
SIMD_BINOP_LIST(SIMD_VISIT_BINOP)
#undef SIMD_VISIT_BINOP

void InstructionSelector::VisitS128Select(Node* node) {
  VisitRRRR(this, kRiscvS128Select, node);
}

#define SIMD_VISIT_SELECT_LANE(Name)                  \
  void InstructionSelector::Visit##Name(Node* node) { \
    VisitRRRR(this, kRiscvS128Select, node);          \
  }
SIMD_VISIT_SELECT_LANE(I8x16RelaxedLaneSelect)
SIMD_VISIT_SELECT_LANE(I16x8RelaxedLaneSelect)
SIMD_VISIT_SELECT_LANE(I32x4RelaxedLaneSelect)
SIMD_VISIT_SELECT_LANE(I64x2RelaxedLaneSelect)
#undef SIMD_VISIT_SELECT_LANE

#define VISIT_SIMD_QFMOP(Name, instruction)           \
  void InstructionSelector::Visit##Name(Node* node) { \
    VisitRRRR(this, instruction, node);               \
  }
VISIT_SIMD_QFMOP(F64x2Qfma, kRiscvF64x2Qfma)
VISIT_SIMD_QFMOP(F64x2Qfms, kRiscvF64x2Qfms)
VISIT_SIMD_QFMOP(F32x4Qfma, kRiscvF32x4Qfma)
VISIT_SIMD_QFMOP(F32x4Qfms, kRiscvF32x4Qfms)
#undef VISIT_SIMD_QFMOP

void InstructionSelector::VisitI32x4DotI16x8S(Node* node) {
  RiscvOperandGenerator g(this);
  InstructionOperand temp = g.TempFpRegister(v16);
  InstructionOperand temp1 = g.TempFpRegister(v17);
  InstructionOperand temp2 = g.TempFpRegister(v30);
  InstructionOperand dst = g.DefineAsRegister(node);
  this->Emit(kRiscvVwmul, temp, g.UseRegister(node->InputAt(0)),
             g.UseRegister(node->InputAt(1)), g.UseImmediate(E16),
             g.UseImmediate(m1));
  this->Emit(kRiscvVcompress, temp2, temp, g.UseImmediate(0b01010101),
             g.UseImmediate(E32), g.UseImmediate(m2));
  this->Emit(kRiscvVcompress, temp1, temp, g.UseImmediate(0b10101010),
             g.UseImmediate(E32), g.UseImmediate(m2));
  this->Emit(kRiscvVaddVv, dst, temp1, temp2, g.UseImmediate(E32),
             g.UseImmediate(m1));
}

namespace {

struct ShuffleEntry {
  uint8_t shuffle[kSimd128Size];
  ArchOpcode opcode;
};

// static const ShuffleEntry arch_shuffles[] = {
//     {{0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23},
//      kRiscvS32x4InterleaveRight},
//     {{8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31},
//      kRiscvS32x4InterleaveLeft},
//     {{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27},
//      kRiscvS32x4PackEven},
//     {{4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31},
//      kRiscvS32x4PackOdd},
//     {{0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27},
//      kRiscvS32x4InterleaveEven},
//     {{4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31},
//      kRiscvS32x4InterleaveOdd},

//     {{0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23},
//      kRiscvS16x8InterleaveRight},
//     {{8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31},
//      kRiscvS16x8InterleaveLeft},
//     {{0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29},
//      kRiscvS16x8PackEven},
//     {{2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31},
//      kRiscvS16x8PackOdd},
//     {{0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29},
//      kRiscvS16x8InterleaveEven},
//     {{2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31},
//      kRiscvS16x8InterleaveOdd},
//     {{6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9},
//      kRiscvS16x4Reverse},
//     {{2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13},
//      kRiscvS16x2Reverse},

//     {{0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23},
//      kRiscvS8x16InterleaveRight},
//     {{8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31},
//      kRiscvS8x16InterleaveLeft},
//     {{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30},
//      kRiscvS8x16PackEven},
//     {{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31},
//      kRiscvS8x16PackOdd},
//     {{0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30},
//      kRiscvS8x16InterleaveEven},
//     {{1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31},
//      kRiscvS8x16InterleaveOdd},
//     {{7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8},
//     kRiscvS8x8Reverse},
//     {{3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12},
//     kRiscvS8x4Reverse},
//     {{1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14},
//      kRiscvS8x2Reverse}};

// bool TryMatchArchShuffle(const uint8_t* shuffle, const ShuffleEntry* table,
//                          size_t num_entries, bool is_swizzle,
//                          ArchOpcode* opcode) {
//   uint8_t mask = is_swizzle ? kSimd128Size - 1 : 2 * kSimd128Size - 1;
//   for (size_t i = 0; i < num_entries; ++i) {
//     const ShuffleEntry& entry = table[i];
//     int j = 0;
//     for (; j < kSimd128Size; ++j) {
//       if ((entry.shuffle[j] & mask) != (shuffle[j] & mask)) {
//         break;
//       }
//     }
//     if (j == kSimd128Size) {
//       *opcode = entry.opcode;
//       return true;
//     }
//   }
//   return false;
// }

}  // namespace

void InstructionSelector::VisitI8x16Shuffle(Node* node) {
  uint8_t shuffle[kSimd128Size];
  bool is_swizzle;
  CanonicalizeShuffle(node, shuffle, &is_swizzle);
  Node* input0 = node->InputAt(0);
  Node* input1 = node->InputAt(1);
  RiscvOperandGenerator g(this);
  // uint8_t shuffle32x4[4];
  // ArchOpcode opcode;
  // if (TryMatchArchShuffle(shuffle, arch_shuffles, arraysize(arch_shuffles),
  //                         is_swizzle, &opcode)) {
  //   VisitRRR(this, opcode, node);
  //   return;
  // }
  // uint8_t offset;
  // if (wasm::SimdShuffle::TryMatchConcat(shuffle, &offset)) {
  //   Emit(kRiscvS8x16Concat, g.DefineSameAsFirst(node), g.UseRegister(input1),
  //        g.UseRegister(input0), g.UseImmediate(offset));
  //   return;
  // }
  // if (wasm::SimdShuffle::TryMatch32x4Shuffle(shuffle, shuffle32x4)) {
  //   Emit(kRiscvS32x4Shuffle, g.DefineAsRegister(node), g.UseRegister(input0),
  //        g.UseRegister(input1),
  //        g.UseImmediate(wasm::SimdShuffle::Pack4Lanes(shuffle32x4)));
  //   return;
  // }
  Emit(kRiscvI8x16Shuffle, g.DefineAsRegister(node), g.UseRegister(input0),
       g.UseRegister(input1),
       g.UseImmediate(wasm::SimdShuffle::Pack4Lanes(shuffle)),
       g.UseImmediate(wasm::SimdShuffle::Pack4Lanes(shuffle + 4)),
       g.UseImmediate(wasm::SimdShuffle::Pack4Lanes(shuffle + 8)),
       g.UseImmediate(wasm::SimdShuffle::Pack4Lanes(shuffle + 12)));
}

void InstructionSelector::VisitI8x16Swizzle(Node* node) {
  RiscvOperandGenerator g(this);
  InstructionOperand temps[] = {g.TempSimd128Register()};
  // We don't want input 0 or input 1 to be the same as output, since we will
  // modify output before do the calculation.
  Emit(kRiscvVrgather, g.DefineAsRegister(node),
       g.UseUniqueRegister(node->InputAt(0)),
       g.UseUniqueRegister(node->InputAt(1)), g.UseImmediate(E8),
       g.UseImmediate(m1), arraysize(temps), temps);
}

void InstructionSelector::VisitSignExtendWord8ToInt32(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvSignExtendByte, g.DefineAsRegister(node),
       g.UseRegister(node->InputAt(0)));
}

void InstructionSelector::VisitSignExtendWord16ToInt32(Node* node) {
  RiscvOperandGenerator g(this);
  Emit(kRiscvSignExtendShort, g.DefineAsRegister(node),
       g.UseRegister(node->InputAt(0)));
}

void InstructionSelector::VisitF32x4Pmin(Node* node) {
  VisitUniqueRRR(this, kRiscvF32x4Pmin, node);
}

void InstructionSelector::VisitF32x4Pmax(Node* node) {
  VisitUniqueRRR(this, kRiscvF32x4Pmax, node);
}

void InstructionSelector::VisitF64x2Pmin(Node* node) {
  VisitUniqueRRR(this, kRiscvF64x2Pmin, node);
}

void InstructionSelector::VisitF64x2Pmax(Node* node) {
  VisitUniqueRRR(this, kRiscvF64x2Pmax, node);
}

#define VISIT_EXT_MUL(OPCODE1, OPCODE2, TYPE)                            \
  void InstructionSelector::Visit##OPCODE1##ExtMulLow##OPCODE2##S(       \
      Node* node) {                                                      \
    RiscvOperandGenerator g(this);                                       \
    Emit(kRiscvVwmul, g.DefineAsRegister(node),                          \
         g.UseUniqueRegister(node->InputAt(0)),                          \
         g.UseUniqueRegister(node->InputAt(1)), g.UseImmediate(E##TYPE), \
         g.UseImmediate(mf2));                                           \
  }                                                                      \
  void InstructionSelector::Visit##OPCODE1##ExtMulHigh##OPCODE2##S(      \
      Node* node) {                                                      \
    RiscvOperandGenerator g(this);                                       \
    InstructionOperand t1 = g.TempFpRegister(v16);                       \
    Emit(kRiscvVslidedown, t1, g.UseUniqueRegister(node->InputAt(0)),    \
         g.UseImmediate(kRvvVLEN / TYPE / 2), g.UseImmediate(E##TYPE),   \
         g.UseImmediate(m1));                                            \
    InstructionOperand t2 = g.TempFpRegister(v17);                       \
    Emit(kRiscvVslidedown, t2, g.UseUniqueRegister(node->InputAt(1)),    \
         g.UseImmediate(kRvvVLEN / TYPE / 2), g.UseImmediate(E##TYPE),   \
         g.UseImmediate(m1));                                            \
    Emit(kRiscvVwmul, g.DefineAsRegister(node), t1, t2,                  \
         g.UseImmediate(E##TYPE), g.UseImmediate(mf2));                  \
  }                                                                      \
  void InstructionSelector::Visit##OPCODE1##ExtMulLow##OPCODE2##U(       \
      Node* node) {                                                      \
    RiscvOperandGenerator g(this);                                       \
    Emit(kRiscvVwmulu, g.DefineAsRegister(node),                         \
         g.UseUniqueRegister(node->InputAt(0)),                          \
         g.UseUniqueRegister(node->InputAt(1)), g.UseImmediate(E##TYPE), \
         g.UseImmediate(mf2));                                           \
  }                                                                      \
  void InstructionSelector::Visit##OPCODE1##ExtMulHigh##OPCODE2##U(      \
      Node* node) {                                                      \
    RiscvOperandGenerator g(this);                                       \
    InstructionOperand t1 = g.TempFpRegister(v16);                       \
    Emit(kRiscvVslidedown, t1, g.UseUniqueRegister(node->InputAt(0)),    \
         g.UseImmediate(kRvvVLEN / TYPE / 2), g.UseImmediate(E##TYPE),   \
         g.UseImmediate(m1));                                            \
    InstructionOperand t2 = g.TempFpRegister(v17);                       \
    Emit(kRiscvVslidedown, t2, g.UseUniqueRegister(node->InputAt(1)),    \
         g.UseImmediate(kRvvVLEN / TYPE / 2), g.UseImmediate(E##TYPE),   \
         g.UseImmediate(m1));                                            \
    Emit(kRiscvVwmulu, g.DefineAsRegister(node), t1, t2,                 \
         g.UseImmediate(E##TYPE), g.UseImmediate(mf2));                  \
  }

VISIT_EXT_MUL(I64x2, I32x4, 32)
VISIT_EXT_MUL(I32x4, I16x8, 16)
VISIT_EXT_MUL(I16x8, I8x16, 8)
#undef VISIT_EXT_MUL

template <unsigned N>
static void VisitInt32PairBinop(InstructionSelector* selector,
                                InstructionCode pair_opcode,
                                InstructionCode single_opcode, Node* node) {
  static_assert(N == 3 or N == 4,
                "Pair operations can only have 3 or 4 inputs");

  RiscvOperandGenerator g(selector);

  Node* projection1 = NodeProperties::FindProjection(node, 1);

  if (projection1) {
    InstructionOperand outputs[] = {
        g.DefineAsRegister(node),
        g.DefineAsRegister(NodeProperties::FindProjection(node, 1))};

    if constexpr (N == 3) {
      // We use UseUniqueRegister here to avoid register sharing with the output
      // register.
      InstructionOperand inputs[] = {g.UseUniqueRegister(node->InputAt(0)),
                                     g.UseUniqueRegister(node->InputAt(1)),
                                     g.UseUniqueRegister(node->InputAt(2))};

      selector->Emit(pair_opcode, 2, outputs, N, inputs);

    } else if constexpr (N == 4) {
      // We use UseUniqueRegister here to avoid register sharing with the output
      // register.
      InstructionOperand inputs[] = {g.UseUniqueRegister(node->InputAt(0)),
                                     g.UseUniqueRegister(node->InputAt(1)),
                                     g.UseUniqueRegister(node->InputAt(2)),
                                     g.UseUniqueRegister(node->InputAt(3))};

      selector->Emit(pair_opcode, 2, outputs, N, inputs);
    }

  } else {
    // The high word of the result is not used, so we emit the standard 32 bit
    // instruction.
    selector->Emit(single_opcode, g.DefineSameAsFirst(node),
                   g.UseRegister(node->InputAt(0)),
                   g.UseRegister(node->InputAt(2)));
  }
}

void InstructionSelector::VisitInt32PairAdd(Node* node) {
  VisitInt32PairBinop<4>(this, kRiscvAddPair, kRiscvAdd, node);
}

void InstructionSelector::VisitInt32PairSub(Node* node) {
  VisitInt32PairBinop<4>(this, kRiscvSubPair, kRiscvSub, node);
}

void InstructionSelector::VisitInt32PairMul(Node* node) {
  VisitInt32PairBinop<4>(this, kRiscvMulPair, kRiscvMul32, node);
}

// Shared routine for multiple shift operations.
static void VisitWord32PairShift(InstructionSelector* selector,
                                 InstructionCode opcode, Node* node) {
  RiscvOperandGenerator g(selector);
  Int32Matcher m(node->InputAt(2));
  InstructionOperand shift_operand;
  if (m.HasResolvedValue()) {
    shift_operand = g.UseImmediate(m.node());
  } else {
    shift_operand = g.UseUniqueRegister(m.node());
  }

  // We use UseUniqueRegister here to avoid register sharing with the output
  // register.
  InstructionOperand inputs[] = {g.UseUniqueRegister(node->InputAt(0)),
                                 g.UseUniqueRegister(node->InputAt(1)),
                                 shift_operand};

  Node* projection1 = NodeProperties::FindProjection(node, 1);

  InstructionOperand outputs[2];
  InstructionOperand temps[1];
  int32_t output_count = 0;
  int32_t temp_count = 0;

  outputs[output_count++] = g.DefineAsRegister(node);
  if (projection1) {
    outputs[output_count++] = g.DefineAsRegister(projection1);
  } else {
    temps[temp_count++] = g.TempRegister();
  }

  selector->Emit(opcode, output_count, outputs, 3, inputs, temp_count, temps);
}

void InstructionSelector::VisitWord32PairShl(Node* node) {
  VisitWord32PairShift(this, kRiscvShlPair, node);
}

void InstructionSelector::VisitWord32PairShr(Node* node) {
  VisitWord32PairShift(this, kRiscvShrPair, node);
}

void InstructionSelector::VisitWord32PairSar(Node* node) {
  VisitWord32PairShift(this, kRiscvSarPair, node);
}

void InstructionSelector::VisitWord32AtomicPairLoad(Node* node) {
  RiscvOperandGenerator g(this);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);
  ArchOpcode opcode = kRiscvWord32AtomicPairLoad;
  AddressingMode addressing_mode = kMode_MRI;
  InstructionCode code = opcode | AddressingModeField::encode(addressing_mode);
  InstructionOperand inputs[] = {g.UseRegister(base), g.UseRegister(index)};
  InstructionOperand temps[3];
  size_t temp_count = 0;
  temps[temp_count++] = g.TempRegister(t0);
  InstructionOperand outputs[2];
  size_t output_count = 0;

  Node* projection0 = NodeProperties::FindProjection(node, 0);
  Node* projection1 = NodeProperties::FindProjection(node, 1);
  if (projection0) {
    outputs[output_count++] = g.DefineAsFixed(projection0, a0);
  } else {
    temps[temp_count++] = g.TempRegister(a0);
  }
  if (projection1) {
    outputs[output_count++] = g.DefineAsFixed(projection1, a1);
  } else {
    temps[temp_count++] = g.TempRegister(a1);
  }
  Emit(code, output_count, outputs, arraysize(inputs), inputs, temp_count,
       temps);
}

void InstructionSelector::VisitWord32AtomicPairStore(Node* node) {
  RiscvOperandGenerator g(this);
  Node* base = node->InputAt(0);
  Node* index = node->InputAt(1);
  Node* value_low = node->InputAt(2);
  Node* value_high = node->InputAt(3);

  InstructionOperand inputs[] = {g.UseRegister(base), g.UseRegister(index),
                                 g.UseFixed(value_low, a1),
                                 g.UseFixed(value_high, a2)};
  InstructionOperand temps[] = {g.TempRegister(a0), g.TempRegister(),
                                g.TempRegister()};
  Emit(kRiscvWord32AtomicPairStore | AddressingModeField::encode(kMode_MRI), 0,
       nullptr, arraysize(inputs), inputs, arraysize(temps), temps);
}

// RV32Gtodo to be implemented
void InstructionSelector::VisitWord32AtomicPairAdd(Node* node) {
  UNIMPLEMENTED();
}

// RV32Gtodo to be implemented
void InstructionSelector::VisitWord32AtomicPairSub(Node* node) {
  UNIMPLEMENTED();
}

// RV32Gtodo to be implemented
void InstructionSelector::VisitWord32AtomicPairAnd(Node* node) {
  UNIMPLEMENTED();
}

// RV32Gtodo to be implemented
void InstructionSelector::VisitWord32AtomicPairOr(Node* node) {
  UNIMPLEMENTED();
}

// RV32Gtodo to be implemented
void InstructionSelector::VisitWord32AtomicPairXor(Node* node) {
  UNIMPLEMENTED();
}

// RV32Gtodo to be implemented
void InstructionSelector::VisitWord32AtomicPairExchange(Node* node) {
  UNIMPLEMENTED();
}

// RV32Gtodo to be implemented
void InstructionSelector::VisitWord32AtomicPairCompareExchange(Node* node) {
  UNIMPLEMENTED();
}

void InstructionSelector::AddOutputToSelectContinuation(OperandGenerator* g,
                                                        int first_input_index,
                                                        Node* node) {
  UNREACHABLE();
}

// static
MachineOperatorBuilder::Flags
InstructionSelector::SupportedMachineOperatorFlags() {
  MachineOperatorBuilder::Flags flags = MachineOperatorBuilder::kNoFlags;
  return flags | MachineOperatorBuilder::kWord32ShiftIsSafe |
         MachineOperatorBuilder::kInt32DivIsSafe |
         MachineOperatorBuilder::kUint32DivIsSafe |
         MachineOperatorBuilder::kFloat32RoundDown |
         MachineOperatorBuilder::kFloat32RoundUp |
         MachineOperatorBuilder::kFloat32RoundTruncate |
         MachineOperatorBuilder::kFloat32RoundTiesEven;
}

// static
MachineOperatorBuilder::AlignmentRequirements
InstructionSelector::AlignmentRequirements() {
#ifdef RISCV_HAS_NO_UNALIGNED
  return MachineOperatorBuilder::AlignmentRequirements::
      NoUnalignedAccessSupport();
#else
  return MachineOperatorBuilder::AlignmentRequirements::
      FullUnalignedAccessSupport();
#endif
}

#undef SIMD_BINOP_LIST
#undef SIMD_SHIFT_OP_LIST
#undef SIMD_UNOP_LIST
#undef SIMD_TYPE_LIST
#undef TRACE_UNIMPL
#undef TRACE

}  // namespace compiler
}  // namespace internal
}  // namespace v8
