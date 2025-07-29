import functools, importlib, re, urllib
from collections import defaultdict
from dataclasses import dataclass
from tinygrad.helpers import getbits, round_up, fetch
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support.usb import ASM24Controller

@dataclass(frozen=True)
class AMDReg:
  name:str; offset:int; segment:int; fields:dict[str, tuple[int, int]]; bases:tuple[int, ...] # noqa: E702

  def encode(self, **kwargs) -> int: return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()), 0)
  def decode(self, val: int) -> dict: return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

  def fields_mask(self, *names) -> int:
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1]-self.fields[nm][0]+1)) - 1) << self.fields[nm][0]) for nm in names), 0)

  @property
  def addr(self): return self.bases[self.segment] + self.offset

@dataclass
class AMDIP:
  name:str; version:tuple[int, ...]; bases:tuple[int, ...] # noqa: E702
  def __post_init__(self): self.version = fixup_ip_version(self.name, self.version)[0]

  @functools.cached_property
  def regs(self): return import_asic_regs(self.name, self.version, cls=functools.partial(AMDReg, bases=self.bases))

  def __getattr__(self, name:str):
    if name in self.regs: return self.regs[name]

    # NOTE: gfx10 gc registers always start with mm, no reg prefix
    return self.regs[name.replace('reg', 'mm')]

def fixup_ip_version(ip:str, version:tuple[int, ...]) -> list[tuple[int, ...]]:
  # override versions
  def _apply_ovrd(ovrd:dict[tuple[int, ...], tuple[int, ...]]) -> tuple[int, ...]:
    for ver, ovrd_ver in ovrd.items():
      if version[:len(ver)] == ver: return ovrd_ver
    return version

  if ip in ['nbio', 'nbif']: version = _apply_ovrd({(3,3): (2,3,0)})
  elif ip in ['mp', 'smu']: version = _apply_ovrd({(14,0,3): (14,0,2)})
  elif ip in ['gc']: version = _apply_ovrd({(9,5,0): (9,4,3)})

  return [version, version[:2], version[:2]+(0,), version[:1]+(0, 0)]

def import_module(name:str, version:tuple[int, ...], version_prefix:str=""):
  for ver in fixup_ip_version(name, version):
    try: return importlib.import_module(f"tinygrad.runtime.autogen.am.{name}_{version_prefix}{'_'.join(map(str, ver))}")
    except ImportError: pass
  raise ImportError(f"Failed to load autogen module for {name.upper()} {'.'.join(map(str, version))}")

def import_asic_regs(prefix:str, version:tuple[int, ...], cls=AMDReg) -> dict[str, AMDReg]:
  def _split_name(name): return name[:(pos:=next((i for i,c in enumerate(name) if c.isupper()), len(name)))], name[pos:]
  def _extract_regs(txt):
    return {m.group(1): int(m.group(2), 0) for line in txt.splitlines() if (m:=re.match(r'#define\s+(\S+)\s+(0x[\da-fA-F]+|\d+)', line))}
  def _download_file(ver, suff) -> str:
    dir_prefix = {"osssys": "oss"}.get(prefix, prefix)
    fetch_name, file_name = f"{prefix}_{'_'.join(map(str, ver))}_{suff}.h", f"{prefix}_{'_'.join(map(str, version))}_{suff}.h"
    url = "https://gitlab.com/linux-kernel/linux-next/-/raw/cf6d949a409e09539477d32dbe7c954e4852e744/drivers/gpu/drm/amd/include/asic_reg"
    return fetch(f"{url}/{dir_prefix}/{fetch_name}", name=file_name, subdir="asic_regs").read_text()

  for ver in fixup_ip_version(prefix, version):
    try: offs, sh_masks = _extract_regs(_download_file(ver, "offset")), _extract_regs(_download_file(ver, "sh_mask"))
    except urllib.error.HTTPError as e:
      if e.code == 404: continue
      raise

    offsets = {k:v for k,v in offs.items() if _split_name(k)[0] in {'reg', 'mm'} and not k.endswith('_BASE_IDX')}
    bases = {k[:-len('_BASE_IDX')]:v for k,v in offs.items() if _split_name(k)[0] in {'reg', 'mm'} and k.endswith('_BASE_IDX')}

    fields: defaultdict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for field_name, field_mask in sh_masks.items():
      if not ('__' in field_name and field_name.endswith('_MASK')): continue
      reg_name, reg_field_name = field_name[:-len('_MASK')].split('__')
      fields[reg_name][reg_field_name.lower()] = ((field_mask & -field_mask).bit_length()-1, field_mask.bit_length()-1)

    # NOTE: Some registers like regGFX_IMU_FUSESTRAP in gc_11_0_0 are missing base idx, just skip them
    return {reg:cls(name=reg, offset=off, segment=bases[reg], fields=fields[_split_name(reg)[1]]) for reg,off in offsets.items() if reg in bases}
  raise ImportError(f"Failed to load ASIC registers for {prefix.upper()} {'.'.join(map(str, version))}")

def setup_pci_bars(usb:ASM24Controller, gpu_bus:int, mem_base:int, pref_mem_base:int) -> dict[int, tuple[int, int]]:
  # Pre-compute constants
  command_flags = pci.PCI_COMMAND_IO | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER
  mem_base_high = (mem_base >> 16) & 0xffff
  pref_mem_base_high = (pref_mem_base >> 16) & 0xffff
  pref_mem_base_upper = pref_mem_base >> 32
  max16 = 0xffff
  max32 = 0xffffffff
  mem_limit_high = max16
  pref_mem_limit_high = max16
  pref_limit_upper = max32

  pcie_cfg_req = usb.pcie_cfg_req  # Reduce attribute lookup overhead

  # Set up buses
  for bus in range(gpu_bus):
    buses = ((bus+1) << 8) | (gpu_bus << 16)
    pcie_cfg_req(pci.PCI_PRIMARY_BUS, bus=bus, dev=0, fn=0, value=buses, size=4)

    pcie_cfg_req(pci.PCI_MEMORY_BASE, bus=bus, dev=0, fn=0, value=mem_base_high, size=2)
    pcie_cfg_req(pci.PCI_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=mem_limit_high, size=2)
    pcie_cfg_req(pci.PCI_PREF_MEMORY_BASE, bus=bus, dev=0, fn=0, value=pref_mem_base_high, size=2)
    pcie_cfg_req(pci.PCI_PREF_MEMORY_LIMIT, bus=bus, dev=0, fn=0, value=pref_mem_limit_high, size=2)
    pcie_cfg_req(pci.PCI_PREF_BASE_UPPER32,  bus=bus, dev=0, fn=0, value=pref_mem_base_upper, size=4)
    pcie_cfg_req(pci.PCI_PREF_LIMIT_UPPER32, bus=bus, dev=0, fn=0, value=pref_limit_upper, size=4)

    pcie_cfg_req(pci.PCI_COMMAND, bus=bus, dev=0, fn=0, value=command_flags, size=1)

  # resize bar 0
  cap_ptr = 0x100
  while cap_ptr:
    hdr = pcie_cfg_req(cap_ptr, bus=gpu_bus, dev=0, fn=0, size=4)
    if pci.PCI_EXT_CAP_ID(hdr) == pci.PCI_EXT_CAP_ID_REBAR:
      cap = pcie_cfg_req(cap_ptr + 0x04, bus=gpu_bus, dev=0, fn=0, size=4)
      new_ctrl = (pcie_cfg_req(cap_ptr + 0x08, bus=gpu_bus, dev=0, fn=0, size=4) & ~0x1F00) | \
                ((int(cap >> 4).bit_length() - 1) << 8)
      pcie_cfg_req(cap_ptr + 0x08, bus=gpu_bus, dev=0, fn=0, value=new_ctrl, size=4)
    cap_ptr = pci.PCI_EXT_CAP_NEXT(hdr)

  mem_space_addr = [mem_base, pref_mem_base]
  bar_off = 0
  bars = {}
  addr_0 = pci.PCI_BASE_ADDRESS_0
  while bar_off < 24:
    cfg = pcie_cfg_req(addr_0 + bar_off, bus=gpu_bus, dev=0, fn=0, size=4)
    bar_mem = bool(cfg & pci.PCI_BASE_ADDRESS_MEM_PREFETCH)
    bar_64 = cfg & pci.PCI_BASE_ADDRESS_MEM_TYPE_64

    if (cfg & pci.PCI_BASE_ADDRESS_SPACE) == pci.PCI_BASE_ADDRESS_SPACE_MEMORY:
      # Save register calculation and attribute lookup
      offs = addr_0 + bar_off
      # - Write all-ones to determine BAR size
      pcie_cfg_req(offs, bus=gpu_bus, dev=0, fn=0, value=max32, size=4)
      lo = pcie_cfg_req(offs, bus=gpu_bus, dev=0, fn=0, size=4) & 0xfffffff0

      if bar_64:
        offs_hi = offs + 4
        pcie_cfg_req(offs_hi, bus=gpu_bus, dev=0, fn=0, value=max32, size=4)
        hi = pcie_cfg_req(offs_hi, bus=gpu_bus, dev=0, fn=0, size=4)
      else:
        hi = 0

      bar_size = ((~(((hi << 32) | lo) & ~0xf)) + 1) & (0xffffffffffffffff if bar_64 else 0xffffffff)

      val_low = mem_space_addr[bar_mem] & 0xffffffff
      pcie_cfg_req(offs, bus=gpu_bus, dev=0, fn=0, value=val_low, size=4)
      if bar_64:
        val_high = mem_space_addr[bar_mem] >> 32
        pcie_cfg_req(offs + 4, bus=gpu_bus, dev=0, fn=0, value=val_high, size=4)
      bars[bar_off // 4] = (mem_space_addr[bar_mem], bar_size)
      mem_space_addr[bar_mem] += round_up(bar_size, 2 << 20)

    bar_off += 8 if bar_64 else 4

  pcie_cfg_req(pci.PCI_COMMAND, bus=gpu_bus, dev=0, fn=0, value=command_flags, size=1)
  return bars
