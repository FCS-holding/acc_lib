---------------------------------------------------------------------
----
---- Accelize 2019
----
---- This is a generated file. Use and modify at your own risk.
---------------------------------------------------------------------
library IEEE;
use IEEE.std_logic_1164.all;

library DRM_0X1003000E00010001_LIBRARY;


entity top_drm_activator_0x1003000e00010001 is
port (
  -- AXI4-Stream Bus clock and reset
  drm_aclk              : in  std_logic;
  drm_arstn             : in  std_logic;
  -- AXI4-Stream Bus from DRM Controller
  drm_to_uip_tready     : out std_logic;
  drm_to_uip_tvalid     : in  std_logic;
  drm_to_uip_tdata      : in  std_logic_vector(31 downto 0);
  -- AXI4-Stream Bus to DRM Controller
  uip_to_drm_tready     : in  std_logic;
  uip_to_drm_tvalid     : out std_logic;
  uip_to_drm_tdata      : out std_logic_vector(31 downto 0);
  -- IP core clock and reset
  ip_core_aclk          : in  std_logic;
  -- IP core interface
  metering_event        : in  std_logic;
  activation_code       : out std_logic_vector(127 downto 0)
);
end entity top_drm_activator_0x1003000e00010001;

architecture top_drm_activator_0x1003000e00010001_rtl of top_drm_activator_0x1003000e00010001 is

  constant POR_DURATION        : integer := 16;

  signal por_shifter           : std_logic_vector(POR_DURATION-1 downto 0) := (others=>'0'); -- init for simulation
  signal i_ip_core_arstn       : std_ulogic;

  signal i_drm_to_uip_tdata    : std_logic_vector(31 downto 0);
  signal i_uip_to_drm_tdata    : std_logic_vector(31 downto 0);

begin

  -----------------
  -- Generate POR
  -----------------

  i_ip_core_arstn <= por_shifter(POR_DURATION-1);

  por_proc: process (ip_core_aclk)
  begin
    if rising_edge(ip_core_aclk) then
      por_shifter(0)                       <= '1';
      por_shifter(POR_DURATION-1 downto 1) <= por_shifter(POR_DURATION-2 downto 0);
    end if;
  end process;


  ---------------------------------
  -- Map activator to AXI4-Stream
  ---------------------------------

  drm_to_uip_tready <= drm_to_uip_tvalid;

  process(ip_core_aclk) begin
    if rising_edge(ip_core_aclk) then
      if drm_arstn = '0' then
        uip_to_drm_tvalid <= '0';
      else
        uip_to_drm_tvalid <= '1';
      end if;
    end if;
  end process;

  i_uip_to_drm_tdata(31 downto 4) <= (others=>'0');

    i_drm_to_uip_tdata  <= drm_to_uip_tdata;

    uip_to_drm_tdata    <= i_uip_to_drm_tdata;


  drm_ip_activator_0x1003000e00010001_inst: entity DRM_0X1003000E00010001_LIBRARY.drm_ip_activator_0x1003000e00010001
  port map (
    DRM_ACLK              => drm_aclk,
    DRM_ARSTN             => drm_arstn,
    DRM_BUS_SLAVE_I_CS    => i_drm_to_uip_tdata(5),
    DRM_BUS_SLAVE_I_CYC   => i_drm_to_uip_tdata(4),
    DRM_BUS_SLAVE_I_ADR   => i_drm_to_uip_tdata(3 downto 2),
    DRM_BUS_SLAVE_I_WE    => i_drm_to_uip_tdata(1),
    DRM_BUS_SLAVE_I_DAT   => i_drm_to_uip_tdata(0 downto 0),
    DRM_BUS_SLAVE_O_ACK   => i_uip_to_drm_tdata(3),
    DRM_BUS_SLAVE_O_STA   => i_uip_to_drm_tdata(1),
    DRM_BUS_SLAVE_O_INTR  => i_uip_to_drm_tdata(2),
    DRM_BUS_SLAVE_O_DAT   => i_uip_to_drm_tdata(0 downto 0),
    IP_CORE_ACLK          => ip_core_aclk,
    IP_CORE_ARSTN         => i_ip_core_arstn,
    DRM_EVENT             => metering_event,
    DRM_ARST              => '0',
    ACTIVATION_CODE_READY => open,
    DEMO_MODE             => open,
    ACTIVATION_CODE       => activation_code
  );

end architecture top_drm_activator_0x1003000e00010001_rtl;
