//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Marcin Copik <mcopik@gmail.com> (Silesian University of Technology)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================
package userinterface;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.DefaultComboBoxModel;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.border.TitledBorder;

import prism.PrismException;
import simulator.gpu.RuntimeDeviceInterface;
import simulator.gpu.RuntimeFrameworkInterface;
import simulator.gpu.opencl.RuntimeOpenCL;

@SuppressWarnings("serial")
public class GUISimulationPlatform extends JPanel
{
	/**
	 * Panel title.
	 */
	private final static String TITLE = "Simulation platform";
	//--------------PLATFORM------------------
	/**
	 * Platform selection label.
	 */
	private JLabel labelPlatform = new JLabel();
	/**
	 * Platform selection label - caption.
	 */
	private final static String LABEL_PLATFORM_CAPTION = "Select platform: ";
	/**
	 * ComboBox with platforms.
	 */
	private JComboBox<String> selectPlatform = new JComboBox<>();
	/**
	 * Hard coded, possible platforms to select from.
	 */
	private final static String AVAILABLE_PLATFORMS[] = { "CPU(default implementation)", "OpenCL" };
	/**
	 * Remember current selected platform - to revert in case of error.
	 */
	private int previousPlatformSelection = 0;
	//--------------DEVICE------------------
	/**
	 * Device selection label.
	 */
	private JLabel labelDevice = new JLabel();
	/**
	 * Device selection label - caption.
	 */
	private final static String LABEL_DEVICE_CAPTION = "Select device: ";
	/**
	 * ComboBox with devices.
	 */
	private JComboBox<String> selectDevice = new JComboBox<>();
	/**
	 * Hard coded possible device for default platform.
	 */
	private DefaultComboBoxModel<String> availableDevices = new DefaultComboBoxModel<>(new String[] { "CPU" });
	//--------------DEVICE INFO------------------
	/**
	 * Device name label - caption.
	 */
	private final static String DEVICE_NAME_CAPTION = "Device: ";
	/**
	 * Device platform label - caption.
	 */
	private final static String DEVICE_PLATFORM_CAPTION = "Platform: ";
	/**
	 * Device supported framework version label - caption.
	 */
	private final static String DEVICE_VERSION_CAPTION = "Supported framework version: ";
	/**
	 * Device name label.
	 */
	private JLabel labelName = new JLabel();
	/**
	 * Device platform label.
	 */
	private JLabel labelPlatformName = new JLabel();
	/**
	 * Device supported framework version label.
	 */
	private JLabel labelVersion = new JLabel();
	/**
	 * Current selected platform.
	 */
	private RuntimeFrameworkInterface currentFramework = null;
	/**
	 * Current selected device.
	 */
	private RuntimeDeviceInterface currentDevice = null;

	public GUISimulationPlatform()
	{
		super(new GridBagLayout());
		this.setBorder(new TitledBorder(TITLE));
		initComponentsSelect(this);
		initComponentsDetails(this);
	}

	/**
	 * Write selected simulation platform to SimulationInformation 
	 * @param info
	 */
	public void setSimulationInfo(SimulationInformation info)
	{
		switch (selectPlatform.getSelectedIndex()) {
		case 0:
			info.setStandardSimulator();
			break;
		case 1:
			if (currentDevice != null) {
				currentFramework.selectDevice(currentDevice);
			} else {
				currentFramework.selectDevice(currentFramework.getMaxFlopsDevice());
			}
			info.setSimulatorFramework(currentFramework);
			break;
		}
	}

	/**
	 * Initialize components from section 'device details/info'
	 * @param panel parent
	 */
	private void initComponentsDetails(JPanel panel)
	{

		initLabel(panel, labelName, DEVICE_NAME_CAPTION, 1, 4, GridBagConstraints.WEST);
		initLabel(panel, labelPlatformName, DEVICE_PLATFORM_CAPTION, 1, 5, GridBagConstraints.WEST);
		initLabel(panel, labelVersion, DEVICE_VERSION_CAPTION, 1, 6, GridBagConstraints.WEST);
	}

	/**
	 * Initialize components from section 'device selection'
	 * @param panel parent
	 */
	private void initComponentsSelect(JPanel panel)
	{

		GridBagConstraints gridBagConstraints;
		//SELECT PLATFORM
		initLabel(panel, labelPlatform, LABEL_PLATFORM_CAPTION, 1, 2, GridBagConstraints.WEST);
		gridBagConstraints = new GridBagConstraints();
		gridBagConstraints.gridx = 3;
		gridBagConstraints.gridy = 2;
		gridBagConstraints.fill = GridBagConstraints.HORIZONTAL;
		selectPlatform.setModel(new DefaultComboBoxModel<String>(AVAILABLE_PLATFORMS));
		selectPlatform.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent evt)
			{
				changeSelectedPlatform(evt);
			}
		});
		panel.add(selectPlatform, gridBagConstraints);

		//SELECT DEVICE
		initLabel(panel, labelDevice, LABEL_DEVICE_CAPTION, 1, 3, GridBagConstraints.WEST);
		gridBagConstraints = new GridBagConstraints();
		gridBagConstraints.gridx = 3;
		gridBagConstraints.gridy = 3;
		gridBagConstraints.fill = GridBagConstraints.HORIZONTAL;
		selectDevice.setEnabled(false);
		selectDevice.setModel(availableDevices);
		selectDevice.addActionListener(new ActionListener()
		{
			public void actionPerformed(ActionEvent evt)
			{
				changeSelectedDevice(evt);
			}
		});
		panel.add(selectDevice, gridBagConstraints);

	}

	/**
	 * Initialize label.
	 * @param panel parent
	 * @param label object to operate
	 * @param caption label caption
	 * @param gridX x position
	 * @param gridY y position
	 * @param gridFill GridBar fill method
	 */
	private void initLabel(JPanel panel, JLabel label, String caption, int gridX, int gridY, int gridFill)
	{
		GridBagConstraints gridBagConstraints = new GridBagConstraints();
		gridBagConstraints.gridx = gridX;
		gridBagConstraints.gridy = gridY;
		gridBagConstraints.anchor = gridFill;
		label.setText(caption);
		panel.add(label, gridBagConstraints);
	}

	/**
	 * Action method for platform selection
	 * @param evt
	 */
	private void changeSelectedPlatform(ActionEvent evt)
	{
		switch (selectPlatform.getSelectedIndex()) {
		case 0:
			selectDevice.setEnabled(false);
			labelName.setText(DEVICE_NAME_CAPTION);
			labelPlatformName.setText(DEVICE_PLATFORM_CAPTION);
			labelVersion.setText(DEVICE_VERSION_CAPTION);
			previousPlatformSelection = 0;
			break;
		case 1:
			selectDevice.setEnabled(true);
			try {
				currentFramework = new RuntimeOpenCL();
				selectDevice.setModel(new DefaultComboBoxModel<String>(currentFramework.getDevicesNames()));
				previousPlatformSelection = 1;
			} catch (PrismException exc) {
				JOptionPane.showMessageDialog(null, exc.getMessage(), "Error!", JOptionPane.ERROR_MESSAGE);
				selectPlatform.setSelectedIndex(previousPlatformSelection);
			}
			break;
		}
	}

	/**
	 * Action method for device selection.
	 * @param evt
	 */
	private void changeSelectedDevice(ActionEvent evt)
	{
		RuntimeDeviceInterface[] devices = currentFramework.getDevices();
		currentDevice = devices[selectDevice.getSelectedIndex()];
		labelName.setText(DEVICE_NAME_CAPTION + currentDevice.getName());
		labelPlatformName.setText(DEVICE_PLATFORM_CAPTION + currentDevice.getPlatformName());
		labelVersion.setText(DEVICE_VERSION_CAPTION + currentDevice.getFrameworkVersion());
	}
}