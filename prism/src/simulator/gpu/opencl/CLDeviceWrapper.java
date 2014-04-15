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
package simulator.gpu.opencl;

import simulator.gpu.RuntimeDeviceInterface;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLPlatform;
import com.nativelibs4java.opencl.JavaCL;

public class CLDeviceWrapper implements RuntimeDeviceInterface
{
	private CLDevice device = null;

	public CLDeviceWrapper(CLDevice device)
	{
		this.device = device;
	}

	public String getName()
	{
		System.out.println(device.getExecutionCapabilities());
		for (String str : device.getExtensions())
			System.out.println(str);
		return device.getName();
	}

	public String getPlatformName()
	{
		return device.getPlatform().getName();
	}

	public String getFrameworkVersion()
	{
		return device.getOpenCLVersion();
	}

	public CLContext createDeviceContext()
	{
		return JavaCL.createContext(null, device);
	}

	public CLPlatform getDevicePlatform()
	{
		return device.getPlatform();
	}

	public CLDevice getDevice()
	{
		return device;
	}

	public String toString()
	{
		return device.toString();
	}

	@Override
	public boolean isCPU()
	{
		return device.getType().contains(CLDevice.Type.CPU);
	}

	@Override
	public boolean isGPU()
	{
		return device.getType().contains(CLDevice.Type.GPU);
	}
}