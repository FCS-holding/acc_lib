---------------------------------------------------------------------
----
---- AUTOGENERATED FILE - DO NOT EDIT
---- DRM SCRIPT VERSION 2.1.0
---- DRM HDK VERSION 4.2.1.0
---- DRM VERSION 4.2.1
---- COPYRIGHT (C) ALGODONE
----
---------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

library DRM_LIBRARY;
use DRM_LIBRARY.DRM_PACKAGE.all;
use DRM_LIBRARY.DRM_INTERFACES_PACKAGE.all;
use DRM_LIBRARY.LICENSE_MASTER_PUF_PACKAGE.all;
use DRM_LIBRARY.DRM_IP_COMPONENTS.all;

entity drm_ip_controller is
  generic (
    SYS_BUS_ADR_BEGIN       : natural := 0;
    SYS_BUS_ADR_SIZE        : natural := 16;
    SYS_BUS_DAT_SIZE        : natural := 32;
    READ_WRITE_MAILBOX_SIZE : natural := 16; -- read write mailbox size in number of 32 bits words
    READ_ONLY_MAILBOX_DATA  : std_logic_vector := x"6B70227B65765F676F697372223A226E2E302E3163722D30222C22325F616E646570797472223A226F646E6164695F6D70222C2275646F72695F74637B3A22646E65762222726F646166223A6E6F636C706D6F636E6974756F632E67222C226D7262696C227972616164223A635F617472706D6F697373652C226E6F6D616E22223A22656C7A63662C22626967697322223A226E007D7D22" -- read only mailbox data (left is the first read only mailbox word)
  );
  port (
    -- AXI4 LITE Slave clock and reset
    SYS_AXI4_ACLK                 : in  std_logic;
    SYS_AXI4_ARSTN                : in  std_logic;
    -- AXI4 LITE Slave address write channel
    SYS_AXI4_BUS_SLAVE_I_AW_VALID : in  std_logic;
    SYS_AXI4_BUS_SLAVE_I_AW_ADDR  : in  std_logic_vector(SYS_BUS_ADR_SIZE-1 downto 0);
    SYS_AXI4_BUS_SLAVE_I_AW_PROT  : in  std_logic_vector(2 downto 0);
    SYS_AXI4_BUS_SLAVE_O_AW_READY : out std_logic;
    -- AXI4 LITE Slave address read channel
    SYS_AXI4_BUS_SLAVE_I_AR_VALID : in  std_logic;
    SYS_AXI4_BUS_SLAVE_I_AR_ADDR  : in  std_logic_vector(SYS_BUS_ADR_SIZE-1 downto 0);
    SYS_AXI4_BUS_SLAVE_I_AR_PROT  : in  std_logic_vector(2 downto 0);
    SYS_AXI4_BUS_SLAVE_O_AR_READY : out std_logic;
    -- AXI4 LITE Slave data write channel
    SYS_AXI4_BUS_SLAVE_I_W_VALID  : in  std_logic;
    SYS_AXI4_BUS_SLAVE_I_W_DATA   : in  std_logic_vector(SYS_BUS_DAT_SIZE-1 downto 0);
    SYS_AXI4_BUS_SLAVE_I_W_STRB   : in  std_logic_vector(SYS_BUS_DAT_SIZE/8-1 downto 0);
    SYS_AXI4_BUS_SLAVE_O_W_READY  : out std_logic;
    -- AXI4 LITE Slave data read channel
    SYS_AXI4_BUS_SLAVE_I_R_READY  : in  std_logic;
    SYS_AXI4_BUS_SLAVE_O_R_VALID  : out std_logic;
    SYS_AXI4_BUS_SLAVE_O_R_DATA   : out std_logic_vector(SYS_BUS_DAT_SIZE-1 downto 0);
    SYS_AXI4_BUS_SLAVE_O_R_RESP   : out std_logic_vector(1 downto 0);
    -- AXI4 LITE Slave write response channel
    SYS_AXI4_BUS_SLAVE_I_B_READY  : in  std_logic;
    SYS_AXI4_BUS_SLAVE_O_B_VALID  : out std_logic;
    SYS_AXI4_BUS_SLAVE_O_B_RESP   : out std_logic_vector(1 downto 0);
    -- Chip dna bus
    CHIP_DNA_VALID                : out std_logic;
    CHIP_DNA                      : out std_logic_vector(63 downto 0);
    -- DRM Bus clock and reset
    DRM_ACLK                      : in  std_logic;
    DRM_ARSTN                     : in  std_logic;
    -- DRM Bus master common socket
    DRM_BUS_MASTER_O_CYC          : out std_logic;
    DRM_BUS_MASTER_O_WE           : out std_logic;
    DRM_BUS_MASTER_O_ADR          : out std_logic_vector(1 downto 0);
    DRM_BUS_MASTER_O_DAT          : out std_logic_vector(0 downto 0);
    -- DRM Bus master ip 0 socket
    DRM_BUS_MASTER_O_CS_0         : out std_logic;
    DRM_BUS_MASTER_I_ACK_0        : in  std_logic;
    DRM_BUS_MASTER_I_STA_0        : in  std_logic;
    DRM_BUS_MASTER_I_INTR_0       : in  std_logic;
    DRM_BUS_MASTER_I_DAT_0        : in  std_logic_vector(0 downto 0);
    -- DRM Bus master ip 1 socket
    DRM_BUS_MASTER_O_CS_1         : out std_logic;
    DRM_BUS_MASTER_I_ACK_1        : in  std_logic;
    DRM_BUS_MASTER_I_STA_1        : in  std_logic;
    DRM_BUS_MASTER_I_INTR_1       : in  std_logic;
    DRM_BUS_MASTER_I_DAT_1        : in  std_logic_vector(0 downto 0);
    -- DRM Bus master ip 2 socket
    DRM_BUS_MASTER_O_CS_2         : out std_logic;
    DRM_BUS_MASTER_I_ACK_2        : in  std_logic;
    DRM_BUS_MASTER_I_STA_2        : in  std_logic;
    DRM_BUS_MASTER_I_INTR_2       : in  std_logic;
    DRM_BUS_MASTER_I_DAT_2        : in  std_logic_vector(0 downto 0);
    -- DRM Bus master ip 3 socket
    DRM_BUS_MASTER_O_CS_3         : out std_logic;
    DRM_BUS_MASTER_I_ACK_3        : in  std_logic;
    DRM_BUS_MASTER_I_STA_3        : in  std_logic;
    DRM_BUS_MASTER_I_INTR_3       : in  std_logic;
    DRM_BUS_MASTER_I_DAT_3        : in  std_logic_vector(0 downto 0)
  );
end entity drm_ip_controller;

architecture drm_ip_controller_RTL of drm_ip_controller is

  -- slave selector type
  type T_SLAVE_SELECTOR is (
    SELECT_NONE,
    SELECT_DNA,
    SELECT_IP_0,
    SELECT_IP_1,
    SELECT_IP_2,
    SELECT_IP_3
  );

  -- drm slaves addresses
  constant C_DRM_DNA_ADR : std_logic_vector(C_DRM_BUS_ADR_SIZE-C_DRM_BUS_ADR_LSB_SIZE-1 downto 0) := std_logic_vector(to_unsigned(0, C_DRM_BUS_ADR_SIZE-C_DRM_BUS_ADR_LSB_SIZE));
  constant C_DRM_IP_ADR_0 : std_logic_vector(C_DRM_BUS_ADR_SIZE-C_DRM_BUS_ADR_LSB_SIZE-1 downto 0) := std_logic_vector(to_unsigned(1, C_DRM_BUS_ADR_SIZE-C_DRM_BUS_ADR_LSB_SIZE));
  constant C_DRM_IP_ADR_1 : std_logic_vector(C_DRM_BUS_ADR_SIZE-C_DRM_BUS_ADR_LSB_SIZE-1 downto 0) := std_logic_vector(to_unsigned(2, C_DRM_BUS_ADR_SIZE-C_DRM_BUS_ADR_LSB_SIZE));
  constant C_DRM_IP_ADR_2 : std_logic_vector(C_DRM_BUS_ADR_SIZE-C_DRM_BUS_ADR_LSB_SIZE-1 downto 0) := std_logic_vector(to_unsigned(3, C_DRM_BUS_ADR_SIZE-C_DRM_BUS_ADR_LSB_SIZE));
  constant C_DRM_IP_ADR_3 : std_logic_vector(C_DRM_BUS_ADR_SIZE-C_DRM_BUS_ADR_LSB_SIZE-1 downto 0) := std_logic_vector(to_unsigned(4, C_DRM_BUS_ADR_SIZE-C_DRM_BUS_ADR_LSB_SIZE));

  -- internal signals
  signal S_DRM_BUS_MASTER_I_DAT  : std_logic_vector(C_DRM_BUS_DAT_SIZE-1 downto 0);
  signal S_DRM_BUS_MASTER_I_ACK  : std_logic;
  signal S_DRM_BUS_MASTER_I_INTR : std_logic;
  signal S_DRM_BUS_MASTER_I_STA  : std_logic_vector(3 downto 0);

  signal S_DRM_BUS_MASTER_O_CYC  : std_logic;
  signal S_DRM_BUS_MASTER_O_WE   : std_logic;
  signal S_DRM_BUS_MASTER_O_ADR  : std_logic_vector(C_DRM_BUS_ADR_SIZE-1 downto 0);
  signal S_DRM_BUS_MASTER_O_DAT  : std_logic_vector(C_DRM_BUS_DAT_SIZE-1 downto 0);

  signal S_DRM_BUS_SLAVE_I_CS    : std_logic;
  signal S_DRM_BUS_SLAVE_I_CYC   : std_logic;
  signal S_DRM_BUS_SLAVE_I_WE    : std_logic;
  signal S_DRM_BUS_SLAVE_I_ADR   : std_logic_vector(C_DRM_BUS_ADR_LSB_SIZE-1 downto 0);
  signal S_DRM_BUS_SLAVE_I_DAT   : std_logic_vector(C_DRM_BUS_DAT_SIZE-1 downto 0);

  signal S_DRM_BUS_SLAVE_O_DAT   : std_logic_vector(C_DRM_BUS_DAT_SIZE-1 downto 0);
  signal S_DRM_BUS_SLAVE_O_ACK   : std_logic;
  signal S_DRM_BUS_SLAVE_O_INTR  : std_logic;

  signal S_SPLITTED_DRM_BUS_MASTER_O_ADR : std_logic_vector(C_DRM_BUS_ADR_SIZE-C_DRM_BUS_ADR_LSB_SIZE-1 downto 0);

  signal S_SLAVE_SELECTOR : T_SLAVE_SELECTOR;

  signal S_DRM_ACLK  : std_logic;
  signal S_DRM_ARSTN : std_logic;

  constant C_drm_ip_controller_DRM_VERSION : std_logic_vector(23 downto 0) := x"040201";

begin

  -- drm controller version check
  assert C_drm_ip_controller_DRM_VERSION(23 downto 8) = C_DRM_MASTER_DRM_VERSION(23 downto 8)
    report "The version of the DRM Controller is not supported by the DRM HDK." severity failure;

  -- drm authenticator version check
  assert C_drm_ip_controller_DRM_VERSION(23 downto 8) = C_DRM_AUTHENTICATOR_DRM_VERSION(23 downto 8)
    report "The version of the DRM Authenticator is not supported by the DRM HDK." severity failure;

  S_DRM_ACLK  <= DRM_ACLK;
  S_DRM_ARSTN <= DRM_ARSTN;

  -- drm controller instantiation
  DRM_CONTROLLER_INSTANCE : DRM_CONTROLLER
    generic map (
      4,
      SYS_BUS_ADR_BEGIN,
      READ_WRITE_MAILBOX_SIZE,
      READ_ONLY_MAILBOX_DATA,
      SYS_BUS_ADR_SIZE,
      SYS_BUS_DAT_SIZE
    )
    port map (
      '1',
      '0',
      S_DRM_ACLK,
      S_DRM_ARSTN,
      S_DRM_BUS_MASTER_I_DAT,
      S_DRM_BUS_MASTER_I_ACK,
      S_DRM_BUS_MASTER_I_INTR,
      S_DRM_BUS_MASTER_I_STA,
      S_DRM_BUS_MASTER_O_CYC,
      S_DRM_BUS_MASTER_O_WE,
      S_DRM_BUS_MASTER_O_ADR,
      S_DRM_BUS_MASTER_O_DAT,
      SYS_AXI4_ACLK,
      SYS_AXI4_ARSTN,
      SYS_AXI4_BUS_SLAVE_I_AW_VALID,
      SYS_AXI4_BUS_SLAVE_I_AW_ADDR,
      SYS_AXI4_BUS_SLAVE_I_AW_PROT,
      SYS_AXI4_BUS_SLAVE_O_AW_READY,
      SYS_AXI4_BUS_SLAVE_I_AR_VALID,
      SYS_AXI4_BUS_SLAVE_I_AR_ADDR,
      SYS_AXI4_BUS_SLAVE_I_AR_PROT,
      SYS_AXI4_BUS_SLAVE_O_AR_READY,
      SYS_AXI4_BUS_SLAVE_I_W_VALID,
      SYS_AXI4_BUS_SLAVE_I_W_DATA,
      SYS_AXI4_BUS_SLAVE_I_W_STRB,
      SYS_AXI4_BUS_SLAVE_O_W_READY,
      SYS_AXI4_BUS_SLAVE_I_R_READY,
      SYS_AXI4_BUS_SLAVE_O_R_VALID,
      SYS_AXI4_BUS_SLAVE_O_R_DATA,
      SYS_AXI4_BUS_SLAVE_O_R_RESP,
      SYS_AXI4_BUS_SLAVE_I_B_READY,
      SYS_AXI4_BUS_SLAVE_O_B_VALID,
      SYS_AXI4_BUS_SLAVE_O_B_RESP
    );

  -- drm dna instantiation
  DRM_DNA_INSTANCE : DRM_DNA
    generic map (
      C_LICENSE_MASTER_PUF_RANDOM_ID_NAME
    )
    port map (
      S_DRM_BUS_SLAVE_I_CS,
      '0',
      S_DRM_ARSTN,
      S_DRM_ACLK,
      S_DRM_BUS_SLAVE_I_CYC,
      S_DRM_BUS_SLAVE_I_WE,
      S_DRM_BUS_SLAVE_I_ADR,
      S_DRM_BUS_SLAVE_I_DAT,
      S_DRM_BUS_SLAVE_O_DAT,
      S_DRM_BUS_SLAVE_O_ACK,
      S_DRM_BUS_SLAVE_O_INTR,
      CHIP_DNA_VALID,
      CHIP_DNA
    );

  -- drm bus slave status link
  S_DRM_BUS_MASTER_I_STA(0) <= DRM_BUS_MASTER_I_STA_0;
  S_DRM_BUS_MASTER_I_STA(1) <= DRM_BUS_MASTER_I_STA_1;
  S_DRM_BUS_MASTER_I_STA(2) <= DRM_BUS_MASTER_I_STA_2;
  S_DRM_BUS_MASTER_I_STA(3) <= DRM_BUS_MASTER_I_STA_3;

	-- drm bus slave address multiplexer
  S_SPLITTED_DRM_BUS_MASTER_O_ADR <= S_DRM_BUS_MASTER_O_ADR(C_DRM_BUS_ADR_SIZE-1 downto C_DRM_BUS_ADR_LSB_SIZE);
  P_SLAVE_SELECT : process(S_SPLITTED_DRM_BUS_MASTER_O_ADR)
  begin
    if S_SPLITTED_DRM_BUS_MASTER_O_ADR = C_DRM_DNA_ADR then
      S_SLAVE_SELECTOR <= SELECT_DNA;
    elsif S_SPLITTED_DRM_BUS_MASTER_O_ADR = C_DRM_IP_ADR_0 then
      S_SLAVE_SELECTOR <= SELECT_IP_0;
    elsif S_SPLITTED_DRM_BUS_MASTER_O_ADR = C_DRM_IP_ADR_1 then
      S_SLAVE_SELECTOR <= SELECT_IP_1;
    elsif S_SPLITTED_DRM_BUS_MASTER_O_ADR = C_DRM_IP_ADR_2 then
      S_SLAVE_SELECTOR <= SELECT_IP_2;
    elsif S_SPLITTED_DRM_BUS_MASTER_O_ADR = C_DRM_IP_ADR_3 then
      S_SLAVE_SELECTOR <= SELECT_IP_3;
    else
      S_SLAVE_SELECTOR <= SELECT_NONE;
    end if;
  end process P_SLAVE_SELECT;

  -- input registers
  P_INPUT_REGISTERS : process(S_DRM_ARSTN, S_DRM_ACLK)
  begin
    if S_DRM_ARSTN = '0' then
      S_DRM_BUS_MASTER_I_DAT <= (others => '0');
      S_DRM_BUS_MASTER_I_ACK <= '0';
      S_DRM_BUS_MASTER_I_INTR <= '0';
    elsif rising_edge(S_DRM_ACLK) then
      S_DRM_BUS_MASTER_I_DAT <= (others => '0');
      S_DRM_BUS_MASTER_I_ACK <= '0';
      S_DRM_BUS_MASTER_I_INTR <= '0';
      case S_SLAVE_SELECTOR is
        when SELECT_DNA =>
          S_DRM_BUS_MASTER_I_DAT <= S_DRM_BUS_SLAVE_O_DAT;
          S_DRM_BUS_MASTER_I_ACK <= S_DRM_BUS_SLAVE_O_ACK;
          S_DRM_BUS_MASTER_I_INTR <= S_DRM_BUS_SLAVE_O_INTR;
        when SELECT_IP_0 =>
          S_DRM_BUS_MASTER_I_DAT <= DRM_BUS_MASTER_I_DAT_0;
          S_DRM_BUS_MASTER_I_ACK <= DRM_BUS_MASTER_I_ACK_0;
          S_DRM_BUS_MASTER_I_INTR <= DRM_BUS_MASTER_I_INTR_0;
        when SELECT_IP_1 =>
          S_DRM_BUS_MASTER_I_DAT <= DRM_BUS_MASTER_I_DAT_1;
          S_DRM_BUS_MASTER_I_ACK <= DRM_BUS_MASTER_I_ACK_1;
          S_DRM_BUS_MASTER_I_INTR <= DRM_BUS_MASTER_I_INTR_1;
        when SELECT_IP_2 =>
          S_DRM_BUS_MASTER_I_DAT <= DRM_BUS_MASTER_I_DAT_2;
          S_DRM_BUS_MASTER_I_ACK <= DRM_BUS_MASTER_I_ACK_2;
          S_DRM_BUS_MASTER_I_INTR <= DRM_BUS_MASTER_I_INTR_2;
        when SELECT_IP_3 =>
          S_DRM_BUS_MASTER_I_DAT <= DRM_BUS_MASTER_I_DAT_3;
          S_DRM_BUS_MASTER_I_ACK <= DRM_BUS_MASTER_I_ACK_3;
          S_DRM_BUS_MASTER_I_INTR <= DRM_BUS_MASTER_I_INTR_3;
        when others =>
          S_DRM_BUS_MASTER_I_DAT <= (others => '0');
          S_DRM_BUS_MASTER_I_ACK <= '0';
          S_DRM_BUS_MASTER_I_INTR <= '0';
      end case;
    end if;
  end process P_INPUT_REGISTERS;

  -- output registers
  P_OUTPUT_REGISTERS : process(S_DRM_ARSTN, S_DRM_ACLK)
  begin
    if S_DRM_ARSTN = '0' then
      S_DRM_BUS_SLAVE_I_CS <= '0';
      S_DRM_BUS_SLAVE_I_CYC <= '0';
      S_DRM_BUS_SLAVE_I_WE <= '0';
      S_DRM_BUS_SLAVE_I_ADR <= (others => '0');
      S_DRM_BUS_SLAVE_I_DAT <= (others => '0');
      DRM_BUS_MASTER_O_CYC <= '0';
      DRM_BUS_MASTER_O_WE <= '0';
      DRM_BUS_MASTER_O_ADR <= (others => '0');
      DRM_BUS_MASTER_O_DAT <= (others => '0');
      DRM_BUS_MASTER_O_CS_0 <= '0';
      DRM_BUS_MASTER_O_CS_1 <= '0';
      DRM_BUS_MASTER_O_CS_2 <= '0';
      DRM_BUS_MASTER_O_CS_3 <= '0';
    elsif rising_edge(S_DRM_ACLK) then
      S_DRM_BUS_SLAVE_I_CS <= '0';
      S_DRM_BUS_SLAVE_I_CYC <= '0';
      S_DRM_BUS_SLAVE_I_WE <= '0';
      S_DRM_BUS_SLAVE_I_ADR <= (others => '0');
      S_DRM_BUS_SLAVE_I_DAT <= (others => '0');
      DRM_BUS_MASTER_O_CYC <= '0';
      DRM_BUS_MASTER_O_WE <= '0';
      DRM_BUS_MASTER_O_ADR <= (others => '0');
      DRM_BUS_MASTER_O_DAT <= (others => '0');
      DRM_BUS_MASTER_O_CS_0 <= '0';
      DRM_BUS_MASTER_O_CS_1 <= '0';
      DRM_BUS_MASTER_O_CS_2 <= '0';
      DRM_BUS_MASTER_O_CS_3 <= '0';
      case S_SLAVE_SELECTOR is
        when SELECT_DNA =>
          S_DRM_BUS_SLAVE_I_CS <= '1';
          S_DRM_BUS_SLAVE_I_CYC <= S_DRM_BUS_MASTER_O_CYC;
          S_DRM_BUS_SLAVE_I_WE <= S_DRM_BUS_MASTER_O_WE;
          S_DRM_BUS_SLAVE_I_ADR <= S_DRM_BUS_MASTER_O_ADR(C_DRM_BUS_ADR_LSB_SIZE-1 downto 0);
          S_DRM_BUS_SLAVE_I_DAT <= S_DRM_BUS_MASTER_O_DAT;
        when SELECT_IP_0 =>
          DRM_BUS_MASTER_O_CS_0 <= '1';
          DRM_BUS_MASTER_O_CYC <= S_DRM_BUS_MASTER_O_CYC;
          DRM_BUS_MASTER_O_WE <= S_DRM_BUS_MASTER_O_WE;
          DRM_BUS_MASTER_O_ADR <= S_DRM_BUS_MASTER_O_ADR(C_DRM_BUS_ADR_LSB_SIZE-1 downto 0);
          DRM_BUS_MASTER_O_DAT <= S_DRM_BUS_MASTER_O_DAT;
        when SELECT_IP_1 =>
          DRM_BUS_MASTER_O_CS_1 <= '1';
          DRM_BUS_MASTER_O_CYC <= S_DRM_BUS_MASTER_O_CYC;
          DRM_BUS_MASTER_O_WE <= S_DRM_BUS_MASTER_O_WE;
          DRM_BUS_MASTER_O_ADR <= S_DRM_BUS_MASTER_O_ADR(C_DRM_BUS_ADR_LSB_SIZE-1 downto 0);
          DRM_BUS_MASTER_O_DAT <= S_DRM_BUS_MASTER_O_DAT;
        when SELECT_IP_2 =>
          DRM_BUS_MASTER_O_CS_2 <= '1';
          DRM_BUS_MASTER_O_CYC <= S_DRM_BUS_MASTER_O_CYC;
          DRM_BUS_MASTER_O_WE <= S_DRM_BUS_MASTER_O_WE;
          DRM_BUS_MASTER_O_ADR <= S_DRM_BUS_MASTER_O_ADR(C_DRM_BUS_ADR_LSB_SIZE-1 downto 0);
          DRM_BUS_MASTER_O_DAT <= S_DRM_BUS_MASTER_O_DAT;
        when SELECT_IP_3 =>
          DRM_BUS_MASTER_O_CS_3 <= '1';
          DRM_BUS_MASTER_O_CYC <= S_DRM_BUS_MASTER_O_CYC;
          DRM_BUS_MASTER_O_WE <= S_DRM_BUS_MASTER_O_WE;
          DRM_BUS_MASTER_O_ADR <= S_DRM_BUS_MASTER_O_ADR(C_DRM_BUS_ADR_LSB_SIZE-1 downto 0);
          DRM_BUS_MASTER_O_DAT <= S_DRM_BUS_MASTER_O_DAT;
        when others =>
          S_DRM_BUS_SLAVE_I_CS <= '0';
          S_DRM_BUS_SLAVE_I_CYC <= '0';
          S_DRM_BUS_SLAVE_I_WE <= '0';
          S_DRM_BUS_SLAVE_I_ADR <= (others => '0');
          S_DRM_BUS_SLAVE_I_DAT <= (others => '0');
          DRM_BUS_MASTER_O_CYC <= '0';
          DRM_BUS_MASTER_O_WE <= '0';
          DRM_BUS_MASTER_O_ADR <= (others => '0');
          DRM_BUS_MASTER_O_DAT <= (others => '0');
          DRM_BUS_MASTER_O_CS_0 <= '0';
          DRM_BUS_MASTER_O_CS_1 <= '0';
          DRM_BUS_MASTER_O_CS_2 <= '0';
          DRM_BUS_MASTER_O_CS_3 <= '0';
      end case;
    end if;
  end process P_OUTPUT_REGISTERS;

end architecture drm_ip_controller_RTL;
