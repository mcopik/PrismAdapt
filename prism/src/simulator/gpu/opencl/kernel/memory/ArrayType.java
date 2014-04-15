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

import prism.Preconditions;
import simulator.gpu.opencl.kernel.expression.Expression;

public class ArrayType implements VariableInterface
{
	private static class ArrayValue implements CLValue
	{
		private VariableInterface type;
		private CLValue[] values = null;

		public ArrayValue(VariableInterface type, CLValue[] values)
		{
			this.type = type;
			this.values = values;
		}

		@Override
		public boolean validateAssignmentTo(VariableInterface type)
		{
			if (type instanceof ArrayType) {
				ArrayType array = (ArrayType) type;
				if (array.length != values.length) {
					return false;
				}
				return values[0].validateAssignmentTo(type);
			} else {
				return false;
			}
		}

		@Override
		public Expression getSource()
		{
			StringBuilder builder = new StringBuilder("{\n");
			for (CLValue value : values) {
				builder.append(value.getSource().toString().replace("\n", " "));
				builder.append(",\n");
			}
			int len = builder.length();
			builder.delete(len - 2, len - 1);
			builder.append("}");
			return new Expression(builder.toString());
		}
	}

	private final VariableInterface varType;
	public final int length;

	public ArrayType(VariableInterface type, int length)
	{
		this.varType = type;
		this.length = length;
	}

	public ArrayType(ArrayType copy)
	{
		this.varType = copy.varType;
		this.length = copy.length;
	}

	@Override
	public String getType()
	{
		return varType.getType() + "*";
	}

	@Override
	public boolean isStructure()
	{
		return false;
	}

	@Override
	public CLVariable accessField(String varName, String fieldName)
	{
		return null;
	}

	@Override
	public boolean isArray()
	{
		return true;
	}

	@Override
	public CLVariable accessElement(CLVariable var, Expression index)
	{
		return new CLVariable(varType, String.format("%s[%s]", var.varName, index.getSource()));
	}

	@Override
	public String declareVar(String varName)
	{
		return String.format("%s %s[%d]", varType.getType(), varName, length);
	}

	public VariableInterface getInternalType()
	{
		return varType;
	}

	public CLValue initializeArray(CLValue[] values)
	{
		Preconditions.checkCondition(values[0].validateAssignmentTo(varType), "Initialization value can't be assigned to array type!");
		return new ArrayValue(varType, values);
	}
}