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
package simulator.gpu.opencl.kernel;

import java.util.List;

import simulator.RandomNumberGenerator;
import simulator.gpu.opencl.kernel.expression.Expression;
import simulator.gpu.opencl.kernel.expression.Include;
import simulator.gpu.opencl.kernel.expression.KernelComponent;
import simulator.gpu.opencl.kernel.memory.CLValue;
import simulator.gpu.opencl.kernel.memory.CLVariable;

import com.nativelibs4java.opencl.CLKernel;

public abstract class PRNGType
{
	protected final String varName;
	protected final List<Include> includes;
	protected final List<CLVariable> additionalArgs;
	protected RandomNumberGenerator random = new RandomNumberGenerator();

	protected PRNGType(String varName, List<Include> includes, List<CLVariable> additionalArgs)
	{
		this.varName = varName;
		this.includes = includes;
		this.additionalArgs = additionalArgs;
	}

	public abstract KernelComponent initializeGenerator();

	public KernelComponent deinitializeGenerator()
	{
		return new Expression("");
	}

	public List<Include> getIncludes()
	{
		return includes;
	}

	public abstract int numbersPerRandomize();

	public abstract KernelComponent randomize() throws KernelException;

	public CLValue getRandomInt(Expression max)
	{
		return getRandomInt(null, max);
	}

	public abstract CLValue getRandomInt(Expression randNumber, Expression max);

	public CLValue getRandomFloat(Expression max)
	{
		return getRandomFloat(null, max);
	}

	public abstract CLValue getRandomFloat(Expression randNumber, Expression max);

	public CLValue getRandomUnifFloat()
	{
		return getRandomUnifFloat(null);
	}

	public abstract CLValue getRandomUnifFloat(Expression randNumber);

	public List<KernelComponent> getAdditionalDefinitions()
	{
		return null;
	}

	public List<CLVariable> getAdditionalInput()
	{
		return additionalArgs;
	}

	public abstract void setKernelArg(CLKernel kernel, int argNumber, int sampleOffset, int globalWorkSize, int localWorkSize);

	public int kernelArgsNumber()
	{
		return 1;
	}
}