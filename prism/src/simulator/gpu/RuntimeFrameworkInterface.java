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
package simulator.gpu;

import java.util.List;

import parser.State;
import parser.ast.Expression;
import parser.ast.ModulesFile;
import parser.ast.PropertiesFile;
import prism.PrismException;
import prism.PrismLog;
import prism.PrismSettings;
import simulator.gpu.automaton.AbstractAutomaton;
import simulator.sampler.Sampler;

public interface RuntimeFrameworkInterface
{
	/**
	 * Supported device types.
	 */
	public enum DeviceType {
		CPU, GPU
	}

	/**
	 * @return name of the framework
	 */
	String getFrameworkName();

	/**
	 * @return name of available platforms
	 */
	String[] getPlatformNames();

	/**
	 * @return number of available platforms
	 */
	int getPlatformNumber();

	/**
	 * @param i number of platform
	 * @return string with detailed description of i-th platform
	 */
	String getPlatformInfo(int i);

	/**
	 * @return array with available devices
	 */
	RuntimeDeviceInterface[] getDevices();

	/**
	 * @return array with device names
	 */
	String[] getDevicesNames();

	/**
	 * Get the device with maximum FLOPS.
	 * @return theoretically "best" device
	 */
	RuntimeDeviceInterface getMaxFlopsDevice();

	/**
	 * Get the device of selected type with maximum FLOPS.
	 * @param type type of device
	 * @return theoretically "best" device
	 */
	RuntimeDeviceInterface getMaxFlopsDevice(DeviceType type);

	/**
	 * Select device used by the framework.
	 * @param device
	 */
	void selectDevice(RuntimeDeviceInterface device);

	/**
	 * Perform simulation.
	 * @param model
	 * @param properties
	 * @return number of generated samples
	 * @throws PrismException
	 */
	int simulateProperty(AbstractAutomaton model, List<Sampler> properties) throws PrismException;

	/**
	 * Set initial state for simulation. When not set, use default values of variables from model.
	 * @param initialState
	 */
	void setInitialState(State initialState);

	/**
	 * Set maximum length of path in sample. When not set, use default hard-coded value.
	 * @param maxPathLength
	 */
	void setMaxPathLength(long maxPathLength);

	/**
	 * Set reference to PrismSettings object. When not set, use default hard-coded values.
	 * @param settings
	 */
	void setPrismSettings(PrismSettings settings);

	/**
	 * Set reference to mainLog. When not set, then an exception will be thrown.
	 * @param mainLog
	 */
	void setMainLog(PrismLog mainLog);

	void setMF(ModulesFile mf);

	void setPF(PropertiesFile pf);

	void setExprs(Expression expr);
}