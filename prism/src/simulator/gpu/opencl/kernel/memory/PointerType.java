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
package simulator.gpu.opencl.kernel.memory;

import simulator.gpu.opencl.kernel.expression.Expression;

public class PointerType implements VariableInterface
{
	private final VariableInterface internalType;

	public PointerType(VariableInterface type)
	{
		this.internalType = type;
	}

	@Override
	public String getType()
	{
		return internalType.getType() + "*";
	}

	@Override
	public boolean isStructure()
	{
		return internalType.isStructure();
	}

	@Override
	public CLVariable accessField(String varName, String fieldName)
	{
		if (!internalType.isStructure()) {
			return null;
		} else {
			return internalType.accessField(String.format("(*%s)", varName), fieldName);
		}
	}

	public VariableInterface getInternalType()
	{
		return internalType;
	}

	@Override
	public boolean isArray()
	{
		return true;
	}

	@Override
	public CLVariable accessElement(CLVariable var, Expression index)
	{
		return new CLVariable(internalType, String.format("%s[%s]", var.varName, index.getSource()));
	}

	@Override
	public String declareVar(String varName)
	{
		return String.format("%s* %s", internalType.getType(), varName);
	}
}